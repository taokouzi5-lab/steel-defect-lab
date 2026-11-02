import { useState } from 'react'

import type { GradCamData } from '../api/types'
import { formatMs } from '../lib/format'

type ViewMode = 'original' | 'heatmap' | 'overlay'

type Props = {
  originalSrc: string | null
  heatmap: GradCamData | null
  heatmapMs: number | null
}

const VIEW_TABS: { id: ViewMode; label: string }[] = [
  { id: 'original', label: '原图' },
  { id: 'heatmap', label: '热力图' },
  { id: 'overlay', label: '叠加' },
]

export function GradCamViewer({ originalSrc, heatmap, heatmapMs }: Props) {
  const [mode, setMode] = useState<ViewMode>(heatmap ? 'overlay' : 'original')
  const [opacity, setOpacity] = useState(0.7)

  const heatmapSrc = heatmap ? `data:image/${heatmap.format};base64,${heatmap.base64}` : null
  const canOverlay = originalSrc !== null && heatmapSrc !== null

  const isTabDisabled = (id: ViewMode): boolean => {
    if (id === 'original') return originalSrc === null
    if (id === 'heatmap') return heatmapSrc === null
    return !canOverlay
  }

  return (
    <div className="gradcam-viewer">
      <div className="gradcam-tabs" role="tablist" aria-label="Grad-CAM 视图">
        {VIEW_TABS.map((tab) => {
          const isActive = mode === tab.id
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={isActive}
              disabled={isTabDisabled(tab.id)}
              className={`gradcam-tab${isActive ? ' is-active' : ''}`}
              onClick={() => setMode(tab.id)}
            >
              {tab.label}
            </button>
          )
        })}
      </div>

      <div className="gradcam-stage">
        {mode === 'original' && originalSrc && (
          <img className="gradcam-img" src={originalSrc} alt="上传的钢材表面原图" />
        )}
        {mode === 'heatmap' && heatmapSrc && (
          <img className="gradcam-img" src={heatmapSrc} alt="Grad-CAM 热力图" />
        )}
        {mode === 'overlay' && canOverlay && (
          <div className="gradcam-overlay-wrap">
            <img className="gradcam-img" src={originalSrc} alt="上传的钢材表面原图" />
            <img
              className="gradcam-img gradcam-heat"
              src={heatmapSrc}
              alt="Grad-CAM 热力图叠加层"
              style={{ opacity }}
            />
          </div>
        )}
        {!heatmap && <p className="gradcam-hint">本次推理未请求 Grad-CAM 热力图</p>}
      </div>

      {mode === 'overlay' && canOverlay && (
        <label className="gradcam-opacity">
          <span>叠加强度</span>
          <input
            type="range"
            min={0.1}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(event) => setOpacity(Number(event.target.value))}
          />
          <span className="mono">{Math.round(opacity * 100)}%</span>
        </label>
      )}

      {heatmapMs !== null && (
        <p className="gradcam-meta mono">Grad-CAM 耗时 {formatMs(heatmapMs)}</p>
      )}
    </div>
  )
}
