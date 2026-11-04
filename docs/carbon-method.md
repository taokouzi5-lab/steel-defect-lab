# DefectLens 碳核算方法

**最后更新：** 2026-11-05
**实现位置：** `backend/app/carbon/estimator.py`、`backend/app/carbon/constants.py`
**运行时披露：** `GET /api/v1/meta`（由同一份常量生成，不会与实现漂移）

## 适用范围

本文描述 DefectLens **推理阶段**的碳排估算：把一次预测消耗的电能换算为
克二氧化碳当量（gCO2eq）。训练阶段的碳排是另一套独立测量，见
[训练碳排](#训练碳排)。

## 公式

```text
gCO2eq = E_per_inference × CI × PUE × CF

E_per_inference = E_methodD_per_1000 / 1000

其中：
  E_per_inference  单次推理电能（kWh）
  E_methodD_per_1000  MACE-F method-D 实测的每 1000 次推理电能（kWh）
  CI               电网碳强度（gCO2eq/kWh）
  PUE              电能使用效率（部署拓扑系数，无量纲）
  CF               仪器校准系数（无量纲）
```

默认参数：`CI = 582`（中国）、`PUE = 2.00`（工厂地面）、`CF = 1.39`
（T3 档）。

### 计算示例：默认模型 resnet18-int8

```text
E_per_inference = 4.574800494862325e-05 kWh / 1000
                = 4.574800494862325e-08 kWh

gCO2eq = 4.574800494862325e-08 × 582 × 2.00 × 1.39
       ≈ 7.40e-05 gCO2eq / 次
```

手机充电等效：`7.40e-05 / 0.1 ≈ 0.00074 s`，即约 **0.74 毫秒**的充电量。
该等效量只是展示口径，见下文常量表的 caveat。

## 常量与出处

| 常量 | 取值 | 出处与提取方式 |
|---|---|---|
| ResNet-50 推理能耗 | `4.574800494862325e-05` kWh / 1000 次 | mace-f `results/E_methodD_resnet_50_emissions.csv`：取最后一行累计 `energy_consumed`。CodeCarbon 3.2.7，1000 次推理，RTX 3070 Ti Laptop 8 GB + i7-12700H |
| ResNet-18 推理能耗 | 回退到 ResNet-50 的 T3 代表值 | ResNet-18 不在 MACE-F method-D 测量矩阵内；`constants.T3_REPRESENTATIVE_MODEL = "resnet_50"`，同一 T3 档（10M–50M 参数） |
| 电网碳强度 `CI`（China） | `582` gCO2eq/kWh | 逐字复制自 mace-f `carbon_benchmark/config.py` 的 `CARBON_INTENSITY`（来源：CO2 Signal API + IPCC 区域平均；Lacoste 2019、Patterson 2022） |
| 默认区域 | China | 产品面向国内制造场景；MACE-F 的测量运行使用英国基准 231 g/kWh，估算时替换为中国电网 |
| `PUE`（factory_floor） | `2.00` | mace-f `config.PUE["factory_floor"]`（Lacoste 2019、Patterson 2022） |
| 校准系数 `CF`（T3） | `1.39` | mace-f `config.CALIBRATION_FACTORS["T3"]`（Fischer 2025 外接功率计方法；CodeCarbon 系统性低估 25–37%） |
| 手机充电等效速率 | `0.1` gCO2eq / 秒 | 由 mace-f `results/analysis_summary.md` 的等效表反推（0.01 g ≈ 0.1 s、0.86 g ≈ 8 s、7.66 g ≈ 75 s）。**Caveat：** 该名义速率在英国电网下约合 1.5 kW，远高于实体 5–20 W 手机充电器；它只是沿用 MACE-F 已发布分析的展示口径，不是物理测量 |
| 分层系数表 | T1 1.58 / T2 1.43 / T3 1.39 / T4 1.34 / T5 1.25 | mace-f `config.CALIBRATION_FACTORS`；未知模型默认 T3 |

补充说明：

- 电网碳强度是区域年度平均值，不是实时读数；实际碳排随电网调度波动。
- CodeCarbon 的 `emissions` 列本身按测量时的基准配置折算；本项目不使用该列
  直接作为最终结果，而是取 `energy_consumed`，再按部署区域与拓扑重新套用
  `CI × PUE × CF`。
- 全部常量都从 `app.carbon.constants` 读取，`GET /api/v1/meta` 会原样披露
  给前端 About 页。

## 训练碳排

训练碳排由 mace-f 的 `CarbonTracker` 调用 CodeCarbon 3.2.7 实测，原始值
按 CodeCarbon 基准配置折算；校准值再乘 T3 系数 1.39。`artifacts/metrics.json`
保存的是**原始值**，`artifacts/checkpoints/*_run.json` 同时保存校准值。

| 模型 | 训练能耗 | 实测碳排（原始） | 校准碳排（×1.39） | 轮次 | 时长 |
|---|---:|---:|---:|---:|---:|
| ResNet-18 | 0.001336 kWh | 0.3085 g | 0.4288 g | 8 | 37.7 s |
| ResNet-50 | 0.001953 kWh | 0.4511 g | 0.6270 g | 6 | 46.0 s |

训练碳排与推理碳排是两个独立边界：训练发生在开发机 GPU 上，推理估算面向
部署容器；不要把两者相加成“模型总碳排”。

## 三条诚实边界

### 1. 同构硬件估算，不是容器实测

method-D 能耗是在开发机（RTX 3070 Ti Laptop GPU、i7-12700H）上用
CodeCarbon 3.2.7 测得的。生产容器是纯 CPU 环境（torch CPU wheels +
onnxruntime），两者不是同一硬件。当前数值是按模型档位迁移的同构硬件估算，
不是对部署容器的能耗测量。前端与 API 都将它标注为估算值，而不是实测值。

### 2. API 服务端碳排不在边界内

对托管型 LLM API，CodeCarbon 只能看到客户端机器，服务端计算不可见；
MACE-F 的 T5 结果因此明确是本地开销下界。DefectLens 在自己的容器内运行
模型，所以估算覆盖的是该容器的推理电能。仍不在边界内的部分包括：

- 训练碳排（已在上表单独报告）；
- 硬件制造与回收的隐含碳（embodied carbon）；
- 网络传输、存储、日志与监控；
- PUE 系数之外的机房制冷、配电损耗等设施开销；
- 客户端设备能耗。

### 3. ResNet-18 使用代表值回退

ResNet-18 未纳入 MACE-F 的 method-D 测量矩阵。为避免伪精确，它的推理碳排
直接使用同档（T3）代表模型 ResNet-50 的实测能耗，因此 API 中两个模型的
`inference_carbon_co2eq_g` 完全相同（均为 7.4×10⁻⁵ g）。响应中的
`inference_carbon_method` 会返回 `tier-representative-fallback`，与
`mace-f-methodD-measured` 明确区分。

## v1.1 容器实测路线图

1. **容器内实测。** 在部署 CPU 容器中接入 CodeCarbon 或 RAPL / 平台能耗
   计数器，在代表性负载（单请求、并发 2、冷启动）下采样。
2. **替换同构迁移值。** 以本机 1000 次请求为口径测量 `energy_consumed`，
   包含空闲基线，分别给出 FP32 与 INT8 的能耗。
3. **实测 PUE。** 用部署现场数据替代 `factory_floor = 2.00` 的档位假设，
   并记录测量边界（是否含制冷与配电）。
4. **消除回退。** 为 ResNet-18 补做 method-D 直接测量，移除 T3 代表值。
5. **披露不确定性。** 多次重复测量，报告置信区间与 p50/p95 能耗分布，
   在 `/api/v1/meta` 中一并返回。

## 参考

- MACE-F 仓库：[github.com/taokouzi5-lab/mace-f](https://github.com/taokouzi5-lab/mace-f)
- CodeCarbon 3.2.7：[github.com/mlco2/codecarbon](https://github.com/mlco2/codecarbon)
- Fischer R. Ground-truthing AI energy consumption. arXiv:2509.22092, 2025.
- Lacoste A, et al. Quantifying the carbon emissions of machine learning. 2019.
- Patterson D, et al. The carbon footprint of ML training will plateau, then
  shrink. IEEE Computer, 2022.
