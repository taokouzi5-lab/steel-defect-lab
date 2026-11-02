import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'

import { AppError, NETWORK_ERROR_MESSAGE, getCarbonMeta } from '../api/client'
import type { CarbonMeta } from '../api/types'
import { formatNumber } from '../lib/format'

const CONSTANT_LABELS: Record<string, string> = {
  country: '电网区域',
  grid_carbon_intensity_g_per_kwh: '电网碳强度（gCO₂eq/kWh）',
  pue: 'PUE（机房能效）',
  default_calibration_factor: '默认校准系数（Fischer 2025）',
  tier_calibration_factors: '分层校准系数',
  phone_charge_gco2eq_per_second: '手机充电等效速率（gCO₂eq/s）',
  measured_inference_energy_kwh_per_1000: '实测推理能耗（kWh / 1000 次）',
}

const METHOD_NOTES: Record<string, string> = {
  'mace-f-methodD-measured':
    'CodeCarbon 3.2.7 方法D 基准推理测量：开发 GPU（RTX 3070 Ti Laptop）上每模型 1000 次推理的整机能耗，取最终累计读数。',
  'tier-representative-fallback':
    '层级代表值回退：未纳入 mace-f 测量矩阵的模型（如 resnet18）使用同层级（T3）代表模型 resnet_50 的实测能耗。',
}

function renderConstantValue(value: unknown): string {
  if (typeof value === 'number') return formatNumber(value)
  if (typeof value === 'string') return value
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return '—'
}

function constantRows(constants: Record<string, unknown>): ReactNode[] {
  const rows: ReactNode[] = []
  for (const [key, value] of Object.entries(constants)) {
    const label = CONSTANT_LABELS[key] ?? key
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      const entries = Object.entries(value as Record<string, unknown>)
      rows.push(<dt key={`${key}-label`}>{label}</dt>)
      rows.push(
        <dd key={`${key}-value`} className="constants-nested">
          {entries.map(([nestedKey, nestedValue]) => (
            <span key={nestedKey} className="constant-chip">
              <span className="mono">{nestedKey}</span>
              <span className="mono">{renderConstantValue(nestedValue)}</span>
            </span>
          ))}
        </dd>,
      )
    } else {
      rows.push(<dt key={`${key}-label`}>{label}</dt>)
      rows.push(
        <dd key={`${key}-value`} className="mono">
          {renderConstantValue(value)}
        </dd>,
      )
    }
  }
  return rows
}

export default function About() {
  const [meta, setMeta] = useState<CarbonMeta | null>(null)
  const [error, setError] = useState<AppError | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    getCarbonMeta()
      .then((payload) => {
        if (!cancelled) setMeta(payload)
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

  return (
    <section aria-labelledby="about-heading">
      <h1 id="about-heading" className="section-title">
        <span className="kicker">Methodology</span>
        方法论
      </h1>
      <p className="page-lede">
        碳成本公式与常量出处、测量边界声明。数据来自 mace-f 框架的实测基准，
        与后端估算器同一数据源（不会漂移）。
      </p>

      {isLoading && (
        <p className="page-status" aria-busy="true">
          正在加载方法声明…
        </p>
      )}
      {error !== null && (
        <p className="page-error" role="alert">
          {error.message}
        </p>
      )}

      {meta !== null && (
        <div className="about-grid">
          <section className="card about-block" aria-labelledby="about-formula">
            <h2 id="about-formula" className="section-title">
              <span className="kicker">Formula</span>
              估算公式
            </h2>
            <p className="formula mono">{meta.formula}</p>
          </section>

          <section className="card about-block" aria-labelledby="about-constants">
            <h2 id="about-constants" className="section-title">
              <span className="kicker">Constants</span>
              常量与出处
            </h2>
            <dl className="constants-list">{constantRows(meta.constants)}</dl>
          </section>

          <section className="card about-block" aria-labelledby="about-boundaries">
            <h2 id="about-boundaries" className="section-title">
              <span className="kicker">Boundaries</span>
              测量边界声明
            </h2>
            <ul className="boundary-list">
              {meta.boundaries.map((boundary) => (
                <li key={boundary}>{boundary}</li>
              ))}
            </ul>
          </section>

          <section className="card about-block" aria-labelledby="about-methods">
            <h2 id="about-methods" className="section-title">
              <span className="kicker">Methods</span>
              测量方法
            </h2>
            <ul className="method-list">
              {meta.methods.map((method) => (
                <li key={method}>
                  <strong className="mono">{method}</strong>
                  <p>{METHOD_NOTES[method] ?? ''}</p>
                </li>
              ))}
            </ul>
          </section>
        </div>
      )}
    </section>
  )
}
