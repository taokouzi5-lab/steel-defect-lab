# DefectLens（钢表面缺陷检测平台）

**支持 Grad-CAM 解释与单次推理碳核算的钢表面缺陷分类系统。**

[![CI](https://github.com/taokouzi5-lab/steel-defect-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/taokouzi5-lab/steel-defect-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-INT8-005CED.svg)](https://onnxruntime.ai/)

## 在线演示

**立即体验：[taokouz1-defectlen.ms.show](https://taokouz1-defectlen.ms.show)**

托管于 ModelScope 创空间（免费 CPU 实例，2 vCPU / 8 GB）。闲置后首次访问
需唤醒实例（冷启动约 30-90 秒），之后上传、Grad-CAM 热力图与碳成本读数全部
运行在容器 CPU 上。空间主页：[modelscope.cn/studios/Taokouz1/defectlen](https://modelscope.cn/studios/Taokouz1/defectlen)。

仓库地址：[github.com/taokouzi5-lab/steel-defect-lab](https://github.com/taokouzi5-lab/steel-defect-lab)

## 功能特性

- **训练对比（ResNet-18 对 ResNet-50）。** 覆盖 NEU-DET 六类缺陷，固定
  80/10/10 划分（1,440/180/180，seed 42），使用 AMP、warmup + cosine
  学习率调度，并按验证集 macro-F1 早停。训练碳排由 MACE-F `CarbonTracker`
  （CodeCarbon 3.2.7）实测。
- **带精度门禁的 INT8 量化。** ONNX opset 17 导出并做数值一致性校验，再做
  静态 INT8 量化。若 top-1 掉点超过 1.5 个百分点，自动回退动态量化；两种
  INT8 变体都未通过门禁时保留 FP32 产物并如实记录失败。结果回写
  `artifacts/metrics.json`。
- **手写 Grad-CAM。** 按 Selvaraju 等（2017）算法作用于 `layer4`，前向与
  反向 hook 成对注册并在 `finally` 中移除。解释目标为已部署 ONNX 模型预测
  的类别，因此热力图不会解释与 API 返回不一致的类别。
- **每次预测都给出碳成本。** 使用 MACE-F 实测推理能耗、中国电网碳强度、
  工厂地面 PUE 与 Fischer-2025 校准系数计算单次推理 gCO2eq。每个响应都
  携带估算值、方法标签与手机充电等效量。
- **安全导向的 API。** 基于 magic bytes 的上传校验（JPEG/PNG）、5 MB 上限、
  4096×4096 像素预算、EXIF 方向校正、全内存处理（应用层不落盘）、
  每客户端 20 次/分钟限流、带 request ID 的 envelope 错误响应。
- **单容器、单源部署。** 生产镜像内构建 React SPA，并由 FastAPI 与
  `/api/v1` 同源托管：非 root 用户、纯 CPU torch、默认单 worker。

## 核心结果

训练在开发机（RTX 3070 Ti Laptop、i7-12700H）上实测；CPU 延迟也在同一
台机器上测量。数据来源：`artifacts/metrics.json` 与
`artifacts/checkpoints/*_run.json`。

| 模型 | 参数量 | 测试准确率 | Macro F1 | 轮次 | 训练碳排（实测 / 校准） | INT8 top-1 掉点 | ONNX 体积（FP32 → INT8） | CPU p50（FP32 → INT8） |
|---|---|---:|---:|---:|---|---:|---|---|
| ResNet-18 | 11.18 M | 99.44% | 99.44% | 8 | 0.31 g / 0.43 g | 0.00 pt | 42.64 → 10.77 MB | 9.32 → 10.14 ms |
| ResNet-50 | 23.52 M | 98.89% | 98.89% | 6 | 0.45 g / 0.63 g | 1.11 pt | 89.64 → 22.91 MB | 21.90 → 20.35 ms |

**负结果如实记录。** 静态 INT8 把每个模型的体积压缩到约 1/4，精度也保持在
门禁内，但在本机 CPU 上**没有带来延迟加速**：ResNet-18 反而变慢
（p50 9.3 → 10.1 ms），ResNet-50 仅小幅改善（21.9 → 20.3 ms，处于运行间
噪声范围内）。这道门禁关注的是精度与产物体积，不是速度。

默认推理碳排：**每次请求 7.4×10⁻⁵ gCO2eq**（中国电网 582 g/kWh、
PUE 2.0、T3 校准系数 1.39）。ResNet-18 使用 T3 代表值回退，因此该数值在
构造上与具体模型无关；详见 [docs/carbon-method.md](docs/carbon-method.md)。

## 快速开始

### 本地单进程

环境要求：Python 3.10+、Node 20+、pip。

```bash
pip install -e "backend/.[dev]"

cd frontend
npm ci
npm run build
cd ..

uvicorn app.main:create_app --factory --app-dir backend --host 0.0.0.0 --port 3000
```

打开 <http://localhost:3000>。API 在同源下托管构建后的 SPA 与 `/api/v1`。

模型资产（`artifacts/onnx`、`artifacts/checkpoints`）已被 gitignore。
缺少这些资产时服务仍可启动：`/api/v1/health` 返回 200，
`/api/v1/models` 将四个由 metrics 得知的模型列为 `available: false`，
`POST /api/v1/predict` 以标准 envelope 返回 `503 model_not_ready`。

### Docker

```bash
docker build -t defectlens .
docker run --rm -p 7860:7860 defectlens
```

打开 <http://localhost:7860>。要启用预测，请从包含 `artifacts/onnx` 与
`artifacts/checkpoints` 的检出目录构建；裸检出会构建同一镜像并以降级模式
运行（由 CI 断言）。

### 重新训练与重新导出（可选）

```bash
pip install -e "training/.[dev]"
python training/scripts/download_weights.py --model resnet18
python training/scripts/train.py --model resnet18
python training/scripts/evaluate.py --model resnet18
python training/scripts/export_onnx.py --model resnet18
python training/scripts/quantize.py --model resnet18
```

该流程要求 NEU-DET 图像位于 `training/data`（未纳入版本控制）。

## 架构

```
浏览器 → 上传校验 → ONNX INT8 → 可选 Grad-CAM → 碳估算 → JSON envelope
```

完整数据流、单容器拓扑、目录结构与关键设计决策见
**[docs/architecture.md](docs/architecture.md)**。

### API 端点

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/v1/health` | 基础设施探针（无 envelope） |
| GET | `/api/v1/models` | 准确率、延迟与碳排元数据 |
| GET | `/api/v1/samples` | 白名单演示样本 |
| GET | `/api/v1/meta` | 碳核算方法披露 |
| POST | `/api/v1/predict` | multipart 图像分类（可选 Grad-CAM） |
| POST | `/api/v1/predict/sample/{id}` | 免上传的演示样本分类 |

## 测试与 CI

- 后端：202 个 pytest 测试（`pytest backend/tests`，覆盖率门槛 70%）。
- 训练：193 个 pytest 测试（覆盖率门槛 60%，CI 中仅用 CPU）。
- 前端：80 个 Vitest 测试；10 个 Playwright E2E 测试（针对构建后的 SPA）。
- CI（`.github/workflows/ci.yml`）：三个 job —— `test`（训练：ruff、
  bandit、pytest）、`backend`（ruff、bandit、pytest）、`docker-build`
  （从裸检出构建生产 Dockerfile，并在 7860 端口断言启动与降级契约）。

## 数据

数据集为 **NEU-DET**（NEU 表面缺陷数据库）：热轧带钢的六类缺陷 ——
crazing、inclusion、patches、pitted_surface、rolled-in_scale、scratches，
每类 300 张灰度图，共 1,800 张。

> K. Song and Y. Yan, "A noise robust method based on completed local
> binary patterns for hot-rolled steel strip surface defects," *Applied
> Surface Science*, vol. 285, pp. 858–864, 2013. Dataset: Northeastern
> Univ., 2013.

## 碳核算方法

估算公式、每个常量的出处、三条明确的边界，以及 v1.1 容器实测路线图见
**[docs/carbon-method.md](docs/carbon-method.md)**。同一份披露由运行时
`GET /api/v1/meta` 提供，确保文档不会与实现脱节。

## 相关项目

- **[mace-f / MACE-F](https://github.com/taokouzi5-lab/mace-f)** ——
  制造业 AI 碳排测量框架；本项目的 method-D 能耗测量、电网碳强度、PUE
  档位与校准系数均来自该框架。

## 许可证

MIT License。Copyright (c) 2026 Chunqi Sun。见
[MIT 许可证文本](https://opensource.org/licenses/MIT)。
