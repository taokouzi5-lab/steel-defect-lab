# DefectLens 架构说明

**最后更新：** 2026-11-05
**入口文件：** `backend/app/main.py`（`create_app`）、`backend/app/asgi.py`（容器入口）、`frontend/src/main.tsx`、`training/scripts/train.py`

## 总览

DefectLens 是一个单进程 FastAPI 服务加 React SPA 的单容器应用：

- 推理走 ONNX Runtime，提供 ResNet-18 / ResNet-50 的 FP32 与静态 INT8
  四个模型 id；
- 解释走加载 PyTorch checkpoint 的手写 Grad-CAM，作用于 `layer4`；
- 碳核算走 MACE-F 常量表与估算器，结果随每次预测返回；
- 除 `/api/v1/health` 外，所有响应统一使用
  `{success, data, error, meta}` envelope，并携带 request ID。

## 请求数据流

```
浏览器（React 19 + Vite SPA）
   │  POST /api/v1/predict   multipart: file, model, gradcam
   ▼
FastAPI（单进程，gunicorn + uvicorn worker）
   │
   ├─ BodySizeLimitMiddleware
   │     Content-Length > 5 MB + 1 MB（multipart 开销）→ 413 envelope，解析前拒绝
   ├─ slowapi 限流（memory://）
   │     每客户端 > 20 次/分钟 → 429 envelope
   ▼
上传校验（backend/app/core/validation.py）
   │  流式读取上限 → JPEG/PNG magic bytes → Pillow verify()
   │  → 4096×4096 像素预算 → EXIF 方向校正 → 转 RGB
   │  应用层全程内存处理，不写文件
   ▼
ONNX Runtime（resnet18 / resnet50 × fp32 / int8）
   │  int8 = 静态量化产物（opset 17）
   │  有界执行器：同时最多 2 个 session.run，单次超时 10 s
   ▼
Top-3 softmax
   │
   ├─ 可选 Grad-CAM（PyTorch checkpoint，FP32）
   │     layer4 · 前向/反向 hook 成对注册并在 finally 中移除
   │     target_class = ONNX 预测类别（解释与已部署预测一致）
   │     每架构一把计算锁（autograd 不可并发）· base64 PNG，最长边 ≤ 1024 px
   ▼
碳估算（backend/app/carbon/estimator.py）
   │  gCO2eq = 单次推理能耗 × 电网碳强度 × PUE × 校准系数
   │  返回 co2eq_g、energy_kwh、method、手机充电等效
   ▼
JSON envelope
   │  {success, data, error, meta{model, request_id}}
   ▼
SPAStaticFiles：/ 提供 frontend/dist，未知路径回退 index.html
```

### 请求路径上的关键约束

| 阶段 | 约束 | 失败响应 |
|---|---|---|
| 请求体预检 | `Content-Length ≤ 5 MB + 1 MB` | 413 `payload_too_large` |
| 限流 | 20 次/分钟/客户端，进程内计数 | 429 `rate_limited` |
| 上传读取 | 流式读取，超过 5 MB 立即停止 | 413 `payload_too_large` |
| 格式识别 | 只认 JPEG `FF D8 FF` / PNG `89 50 4E 47`，不信任扩展名 | 415 `unsupported_media_type` |
| 图像解码 | `verify()` + 重新打开 + 4096² 像素预算，先于完整解码 | 422 `unprocessable_image` |
| 推理 | 模型白名单、10 s 超时、并发上限 2 | 404 / 503 / 504 envelope |
| Grad-CAM | 需要对应架构 checkpoint；按架构串行 | 503 `model_not_ready` |

## 单容器部署拓扑

```
                         ┌───────────────────────────────────────────────────┐
                         │ 容器（Linux/amd64，非 root，uid 10001）            │
                         │                                                   │
   :7860  ─────────────► │  gunicorn（默认 1 worker，WORKERS 可覆盖）          │
                         │    └─ uvicorn worker                              │
                         │         └─ FastAPI app（app.asgi:app）             │
                         │              ├─ /api/v1/*   REST 端点              │
                         │              └─ /           SPA（frontend/dist）   │
                         │                                                   │
                         │  /app/artifacts/onnx/         ONNX 模型（构建时烘焙）│
                         │  /app/artifacts/checkpoints/  Grad-CAM checkpoint  │
                         │  运行期 /app 只读（PYTHONDONTWRITEBYTECODE=1）      │
                         └───────────────────────────────────────────────────┘
```

- **镜像构建**：`node:24-alpine` 阶段执行 `npm ci && npm run build`；
  `python:3.11-slim` 阶段安装 CPU 版 torch/torchvision 与
  `onnxruntime>=1.21.2`，再复制后端、前端产物与 `artifacts/`。
- **端口**：默认 `0.0.0.0:7860`（ModelScope Docker 约定），可用 `PORT` 覆盖。
- **进程**：`gunicorn --config backend/gunicorn.conf.py app.asgi:app`；
  默认 1 个 worker，理由见下文设计决策。
- **健康探针**：`GET /api/v1/health` 在进程存活期间始终返回 200；
  `models_loaded` 报告启动时预加载的 ONNX session 数量。
- **单源**：SPA 与 API 同源，生产镜像设置 `CORS_ORIGINS=""` 关闭 CORS。

### 无模型资产时的降级契约

`artifacts/onnx` 与 `artifacts/checkpoints` 被 gitignore，但会随构建上下文
（本地检出或 ModelScope 上传）烘焙进镜像。裸检出构建出的镜像仍然可用：

- `/api/v1/health` 返回 200，`models_loaded` 为空；
- `/api/v1/models` 列出四个由 metrics 得知的模型，全部 `available: false`；
- `POST /api/v1/predict` 返回 `503 model_not_ready` envelope。

CI 的 `docker-build` job 用真实生产 Dockerfile 从裸检出构建并逐条断言上述
契约，避免出现与生产镜像漂移的 CI 专用变体。

## 目录结构

```text
steel-defect-lab/
├── backend/
│   ├── app/
│   │   ├── api/            # 路由与依赖注入
│   │   ├── carbon/         # 常量、估算器、/meta 披露
│   │   ├── core/           # envelope、校验、请求体上限、限流、SPA 静态托管
│   │   ├── explain/        # 手写 Grad-CAM 与热力图叠加
│   │   ├── inference/      # ONNX engine、模型注册表
│   │   ├── services/       # 预测编排、Grad-CAM 后端、样本目录
│   │   ├── schemas/        # Pydantic 响应模型
│   │   ├── config.py       # 环境变量配置（无硬编码路径）
│   │   ├── main.py         # create_app：中间件、异常处理、静态托管
│   │   └── asgi.py         # 容器入口
│   ├── gunicorn.conf.py
│   └── tests/              # 202 个 pytest 测试
├── frontend/
│   └── src/
│       ├── pages/          # Home（推理）、Samples、Compare、About
│       ├── components/     # 上传区、置信度条、Grad-CAM、碳排面板等
│       ├── api/            # 类型化客户端
│       ├── hooks/          # usePredict
│       └── lib/            # 格式化与上传预检
├── training/
│   ├── configs/            # resnet18.yaml、resnet50.yaml
│   ├── scripts/            # download_weights、train、evaluate、export_onnx、quantize
│   ├── steel_training/     # data / engine / eval / models
│   ├── tests/              # 193 个 pytest 测试
│   └── manifest.json       # seed 42，1440/180/180 划分
├── artifacts/
│   ├── metrics.json        # 训练、量化与延迟指标（纳入版本控制）
│   ├── carbon/             # 训练碳排 CSV
│   ├── onnx/               # FP32 / INT8 模型（gitignore）
│   └── checkpoints/        # 训练 checkpoint（gitignore）
├── docs/
│   ├── architecture.md     # 本文
│   ├── carbon-method.md    # 碳核算方法与边界
│   └── deployment.md       # 部署参数与运维说明
├── Dockerfile              # 单容器生产镜像
└── .github/workflows/ci.yml
```

## 关键设计决策

| 决策 | 原因 | 代价 / 备注 |
|---|---|---|
| **手写 Grad-CAM**，不引入 `pytorch-grad-cam` | 依赖面小、算法可审计；hook 注册与移除严格配对（`finally`），无跨请求状态泄漏；`target_class` 取自 ONNX 预测，量化翻转近分时热力图仍与 API 返回一致 | 需自行维护约 200 行实现，并加每架构计算锁保证 autograd 串行 |
| **静态 INT8 量化 + 精度门禁** | 部署目标是纯 CPU；INT8 将模型体积压缩约 4×，门禁（top-1 掉点 ≤ 1.5 pt）防止为体积牺牲精度；失败自动回退动态量化，再失败保留 FP32 | 本机 CPU 上未获得延迟收益：ResNet-18 变慢，ResNet-50 仅噪声级改善。门禁不以速度为指标，负结果如实记录 |
| **默认单 worker** | slowapi 使用 `memory://`，计数在进程内；N 个 worker 会把每客户端预算放大 N 倍。同时 ONNX session 缓存、有界推理执行器与 Grad-CAM checkpoint 缓存只保留一份 | 提高吞吐需要共享限流后端；`WORKERS` 可覆盖，但需同步更换限流存储 |
| **ResNet-18 的推理碳用 T3 代表值回退** | ResNet-18 不在 MACE-F method-D 测量矩阵内；它属于 T3 档（10M–50M 参数），使用同档 ResNet-50 的实测能耗，比凭空编造测量值更诚实 | API 以 `method: "tier-representative-fallback"` 明示；两个模型的推理碳数值因此相同 |
| **上传全内存处理** | 应用层不落盘，消除路径遍历与残留文件面；大小上限与像素预算限制解码成本 | Starlette multipart 解析在超过约 1 MB 时可能先落临时文件，故用 Content-Length 预检拦截；chunked 请求仍是残余风险（见 `docs/deployment.md`） |
| **SPA 由 API 容器托管** | 单容器、单源，无需额外 Web 服务器；前端在镜像内构建，CI 验证的就是生产 Dockerfile | 静态资源与 API 共享进程资源；大流量静态资源应交由 CDN/网关 |

## 相关文档

- [碳核算方法与边界](carbon-method.md)
- [部署指南](deployment.md)
- [英文 README](../README.en.md) · [中文 README](../README.zh.md)
