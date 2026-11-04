# E2E 测试（Playwright · T4.4b）

针对 DefectLens 关键用户流程的端到端测试。被测应用为**单进程服务器**：后端（FastAPI）已内置
SPA 静态托管（`backend/app/core/spa_static.py`），测试全部走 `http://localhost:3000`。

## 前置条件

### 1. 构建前端（生成 dist）

```powershell
cd frontend
npm run build
```

### 2. 启动服务器（测试前手动起，Playwright config 不自动起服务）

```powershell
cd d:\claudecodetest\steel-defect-lab
D:\anaconda\anaconda\python.exe -m uvicorn app.main:create_app --factory --app-dir backend --port 3000
```

启动后确认：

```powershell
curl http://localhost:3000/api/v1/health
# {"status":"ok","models_loaded":[...]}
```

> 速率限制说明：后端默认 `RATE_LIMIT_PER_MIN=20`（按客户端 IP 聚合，含页面加载的
> GET 请求）。单次全新跑一轮约消耗 12 个配额，可正常通过；**连续多轮回归**建议关闭限流
> 以避免 429 抖动：
>
> ```powershell
> $env:RATE_LIMIT_ENABLED="0"
> D:\anaconda\anaconda\python.exe -m uvicorn app.main:create_app --factory --app-dir backend --port 3000
> ```

### 3. 安装浏览器（国内镜像，失败时回退官方 CDN）

```powershell
cd frontend
$env:PLAYWRIGHT_DOWNLOAD_HOST="https://npmmirror.com/mirrors/playwright/"
npx playwright install chromium
```

> 若镜像主机不可达（请求超时）或下载失败，去掉环境变量直接走官方 CDN 重试：
>
> ```powershell
> Remove-Item Env:PLAYWRIGHT_DOWNLOAD_HOST
> npx playwright install chromium
> ```
>
> 两种方式产物一致（`%LOCALAPPDATA%\ms-playwright\chromium-*`），按本机网络择优。

## 运行

```powershell
cd frontend
npm run test:e2e            # 全量（单 worker，串行）
npm run test:e2e -- --headed       # 有头模式观察
npm run test:e2e -- e2e/predict.spec.ts   # 单文件
npx playwright show-report e2e/playwright-report   # HTML 报告
```

## 测试清单

| 文件 | 流程 | 关键断言 |
|------|------|---------|
| `predict.spec.ts` | 上传预测主流程 | 结果面板：分类名非空、置信度 0-100%、推理延迟 > 0、CarbonPanel 出现 CO₂eq、Grad-CAM 出现 `img[alt]`；空态按钮禁用 |
| `samples.spec.ts` | 示例推演 | 12 张卡片、点击第一张 → 结果面板 + 置信度 + 卡片高亮 |
| `errors.spec.ts` | 错误上传 | 伪造 jpg（文本改名）→ 服务端 415 → 友好中文「格式」提示（`role="alert"`）、页面不崩溃、重试可恢复 |
| `responsive.spec.ts` | 截图回归 | 320/768/1440 对 `/`（含预测结果）与 `/samples` 截图；无横向溢出（scrollWidth ≤ innerWidth + 1），违规在截图全部落盘后统一断言 |
| `a11y.spec.ts` | 键盘可达性 | Tab 聚焦 4 导航链接与上传按钮；UploadZone 按钮语义 + Enter 触发文件选择；`role="alert"` / `aria-live` / radiogroup / tablist |

## 测试资产

- 合法钢材图：`backend/tests/fixtures/valid_jpeg.jpg`（已入库，测试直接引用，勿移动）
- 伪造 jpg / 非法扩展名文件：由 `helpers.ts` 在系统临时目录**运行时程序生成**，不入库、OS 自动回收

## 视觉回归基线（screenshots/）

`screenshots/` 下的 6 张 PNG（home/samples × 320/768/1440，fullPage）**已入库**，
作为视觉回归基线：

- **CI / 常规运行**：`responsive.spec.ts` 每次运行都会覆盖这些文件并断言无横向溢出。
  将运行后的 git diff 视为视觉变更审查点（改动符合预期则提交更新，异常则排查）。
- **手动更新基线**：确认 UI 变更为预期后，重跑一次并提交截图：

```powershell
npm run test:e2e -- e2e/responsive.spec.ts
git add e2e/screenshots
```

断言使用结构断言（无横向溢出），不依赖像素比对；像素级差异通过 git diff 人工确认。

## 工程约定

- 单 worker：后端推理执行器并发上限 2，串行避免互相抢占
- 无 `console.log`；TS strict；选择器优先语义化 class / ARIA 属性
- 不硬编码具体缺陷类别（模型资产更新后输出可能变化），只断言结构与数值范围
- `playwright-report/`、`test-results/` 为运行产物，已在 `.gitignore` 中忽略
