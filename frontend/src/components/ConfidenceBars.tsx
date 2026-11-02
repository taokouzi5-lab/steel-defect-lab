import type { CSSProperties } from 'react'

import type { TopKItem } from '../api/types'
import { clamp01, classLabel, formatPercent } from '../lib/format'

type Props = {
  items: TopKItem[]
  highlightClass?: string
}

export function ConfidenceBars({ items, highlightClass }: Props) {
  const sorted = [...items].sort((a, b) => b.confidence - a.confidence)
  return (
    <ul className="confidence-bars">
      {sorted.map((item) => {
        const scale = clamp01(item.confidence)
        const isTop = item.class_name === highlightClass
        const fillStyle = { '--fill-scale': String(scale) } as CSSProperties
        return (
          <li key={item.class_id} className={`confidence-row${isTop ? ' is-top' : ''}`}>
            <span className="confidence-label">{classLabel(item.class_name)}</span>
            <span className="confidence-track">
              <span className="confidence-fill" style={fillStyle} />
            </span>
            <span className="confidence-value mono">{formatPercent(item.confidence)}</span>
          </li>
        )
      })}
    </ul>
  )
}
