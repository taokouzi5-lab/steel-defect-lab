import { useEffect, useState } from 'react'

import { AppError, NETWORK_ERROR_MESSAGE, getModels } from '../api/client'
import type { ModelInfo } from '../api/types'
import { formatCarbon, formatMs, formatParams, formatPercent } from '../lib/format'

const METHOD_BADGES: Record<string, string> = {
  'mace-f-methodD-measured': '实测',
  'tier-representative-fallback': 'T3 代表值',
}

function byAccuracyThenCarbon(a: ModelInfo, b: ModelInfo): number {
  if (b.test_acc !== a.test_acc) return b.test_acc - a.test_acc
  return a.inference_carbon_co2eq_g - b.inference_carbon_co2eq_g
}

export default function Compare() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [error, setError] = useState<AppError | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    getModels()
      .then((list) => {
        if (!cancelled) setModels([...list].sort(byAccuracyThenCarbon))
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof AppError ? err : new AppError('network', NETWORK_ERROR_MESSAGE))
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const recommendedId = models.length > 0 ? models[0].id : null

  return (
    <section aria-labelledby="compare-heading">
      <h1 id="compare-heading" className="section-title">
        <span className="kicker">Model Selection</span>
        模型对比
      </h1>
      <p className="page-lede">
        选型标准：精度达标前提下碳效率最优。数据来自训练实测与 INT8 量化门禁，
        推理碳为单次推理的碳估算（详见方法论页）。
      </p>

      {isLoading && (
        <p className="page-status" aria-busy="true">
          正在加载模型基准…
        </p>
      )}
      {error !== null && (
        <p className="page-error" role="alert">
          {error.message}
        </p>
      )}

      {models.length > 0 && (
        <div className="compare-table-wrap">
          <table className="compare-table">
            <caption className="compare-caption mono">
              DEFECTLENS / 选型决策表 · 碳效率列已高亮
            </caption>
            <thead>
              <tr>
                <th scope="col">模型</th>
                <th scope="col">精度</th>
                <th scope="col">参数量</th>
                <th scope="col">CPU 延迟（p50 / p95）</th>
                <th scope="col" className="col-carbon">
                  推理碳
                </th>
                <th scope="col">训练碳</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => {
                const inference = formatCarbon(model.inference_carbon_co2eq_g)
                const training = formatCarbon(model.training_carbon_co2eq_g)
                const isRecommended = model.id === recommendedId
                return (
                  <tr key={model.id} className={isRecommended ? 'is-recommended' : undefined}>
                    <th scope="row">
                      <span className="mono">{model.id}</span>
                      <span className="model-cell-sub mono">
                        {model.onnx_size_mb.toFixed(1)} MB · {METHOD_BADGES[model.inference_carbon_method] ?? model.inference_carbon_method}
                      </span>
                      {isRecommended && <span className="recommend-badge">碳效率最优</span>}
                    </th>
                    <td className="mono">{formatPercent(model.test_acc)}</td>
                    <td className="mono">{formatParams(model.params)}</td>
                    <td className="mono">
                      {formatMs(model.cpu_latency_ms.p50)} / {formatMs(model.cpu_latency_ms.p95)}
                    </td>
                    <td className="mono col-carbon">
                      {inference.text} {inference.unit}
                    </td>
                    <td className="mono">
                      {training.text} {training.unit}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      <p className="compare-note">
        注：resnet18 未纳入 mace-f 测量矩阵，其推理碳采用 T3 层级代表模型（resnet_50）的实测能耗回退估算；
        resnet50 系列为 CodeCarbon 方法D 直接测量。
      </p>
    </section>
  )
}
