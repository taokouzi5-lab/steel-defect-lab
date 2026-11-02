import { Link } from 'react-router-dom'

import type { CarbonData } from '../api/types'
import { carbonEquivalentText, formatCarbon, formatNumber } from '../lib/format'

type Props = {
  carbon: CarbonData
}

const METHOD_LABELS: Record<string, string> = {
  'mace-f-methodD-measured': 'mace-f 方法D 实测',
  'tier-representative-fallback': 'T3 层级代表值回退',
}

export function CarbonPanel({ carbon }: Props) {
  const formatted = formatCarbon(carbon.co2eq_g)
  const methodLabel = METHOD_LABELS[carbon.method] ?? carbon.method
  return (
    <section className="carbon-panel" aria-label="本次推理碳成本">
      <header className="carbon-head">
        <h3>
          <span className="kicker">Carbon</span>
          碳成本
        </h3>
        <span className="carbon-value">
          <strong className="mono">{formatted.text}</strong>
          <span className="carbon-unit">
            {formatted.unit} CO₂eq
          </span>
        </span>
      </header>
      <p className="carbon-equivalent">{carbonEquivalentText(carbon.co2eq_g, carbon.equivalent)}</p>
      <dl className="carbon-details">
        <div>
          <dt>估算方法</dt>
          <dd>{methodLabel}</dd>
        </div>
        <div>
          <dt>推理能耗</dt>
          <dd className="mono">{formatNumber(carbon.energy_kwh)} kWh</dd>
        </div>
      </dl>
      <Link className="carbon-method-link" to="/about">
        碳估算方法论 →
      </Link>
    </section>
  )
}
