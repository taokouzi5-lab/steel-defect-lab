import { useCallback, useEffect, useRef, useState } from 'react'

import { AppError, DEFAULT_MODEL_ID, NETWORK_ERROR_MESSAGE, getModels } from '../api/client'
import type { ModelInfo } from '../api/types'
import { CarbonPanel } from '../components/CarbonPanel'
import { ConfidenceBars } from '../components/ConfidenceBars'
import { GradCamViewer } from '../components/GradCamViewer'
import { ModelPicker } from '../components/ModelPicker'
import { UploadZone } from '../components/UploadZone'
import { usePredict } from '../hooks/usePredict'
import { classLabel, formatCarbon, formatMs, formatPercent } from '../lib/format'

interface RecentEntry {
  id: string
  className: string
  confidence: number
  co2eqG: number
  model: string
}

const MAX_RECENT = 5

export default function Home() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [modelsError, setModelsError] = useState<AppError | null>(null)
  const [modelsLoading, setModelsLoading] = useState(true)
  const [modelsReloadTick, setModelsReloadTick] = useState(0)
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL_ID)
  const [wantGradcam, setWantGradcam] = useState(true)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [recent, setRecent] = useState<RecentEntry[]>([])
  const { phase, data, error, run } = usePredict()
  const previewRef = useRef<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getModels()
      .then((list) => {
        if (cancelled) return
        setModels(list.filter((model) => model.available))
        setModelsError(null)
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setModelsError(
            err instanceof AppError ? err : new AppError('network', NETWORK_ERROR_MESSAGE),
          )
        }
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [modelsReloadTick])

  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    }
  }, [])

  const handleSelect = useCallback((next: File) => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    const url = URL.createObjectURL(next)
    previewRef.current = url
    setFile(next)
    setPreviewUrl(url)
  }, [])

  const handleClear = useCallback(() => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    previewRef.current = null
    setFile(null)
    setPreviewUrl(null)
  }, [])

  const handlePredict = useCallback(async () => {
    if (!file || phase === 'uploading') return
    const result = await run({ kind: 'file', file }, { model: selectedModel, gradcam: wantGradcam })
    if (result) {
      const entry: RecentEntry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        className: result.class_name,
        confidence: result.confidence,
        co2eqG: result.carbon.co2eq_g,
        model: selectedModel,
      }
      setRecent((prev) => [entry, ...prev].slice(0, MAX_RECENT))
    }
  }, [file, phase, run, selectedModel, wantGradcam])

  const showResult = phase === 'done' && data !== null

  return (
    <section aria-labelledby="home-heading">
      <h1 id="home-heading" className="home-hero">
        钢表面缺陷检测
        <span className="mono home-accent"> · 碳意识选型</span>
      </h1>
      <p className="home-lede">
        上传一张钢材表面图像，resnet18-int8 在本地 CPU 上完成分类，Grad-CAM
        指出缺陷位置，并告诉你这次推理的碳成本。
      </p>

      <ModelPicker
        models={models}
        selectedId={selectedModel}
        isLoading={modelsLoading}
        error={modelsError}
        onSelect={setSelectedModel}
        onRetry={() => {
          setModelsLoading(true)
          setModelsReloadTick((tick) => tick + 1)
        }}
      />

      <div className="inference-grid">
        <div className="inference-col">
          <UploadZone
            onSelect={handleSelect}
            onClear={handleClear}
            previewUrl={previewUrl}
            isUploading={phase === 'uploading'}
          />
          <div className="predict-controls">
            <label className="gradcam-toggle">
              <input
                type="checkbox"
                checked={wantGradcam}
                onChange={(event) => setWantGradcam(event.target.checked)}
              />
              <span>Grad-CAM 热力图</span>
            </label>
            <button
              type="button"
              className="predict-button"
              disabled={!file || phase === 'uploading'}
              onClick={() => void handlePredict()}
            >
              {phase === 'uploading' ? '推理中…' : '开始推理'}
            </button>
          </div>
          {phase === 'error' && error !== null && (
            <p className="predict-error" role="alert">
              {error.message}
            </p>
          )}
        </div>

        <div className="inference-col">
          {showResult && data ? (
            <div className="result-panel">
              <header className="result-head">
                <h2>{classLabel(data.class_name)}</h2>
                <span className="result-confidence mono">{formatPercent(data.confidence)}</span>
                <span className="result-latency mono">
                  推理 {formatMs(data.latency_ms)} · Grad-CAM{' '}
                  {data.gradcam_ms !== null ? formatMs(data.gradcam_ms) : '—'}
                </span>
              </header>
              <ConfidenceBars items={data.top_k} highlightClass={data.class_name} />
              <div className="result-split">
                <GradCamViewer
                  originalSrc={previewUrl}
                  heatmap={data.gradcam}
                  heatmapMs={data.gradcam_ms}
                />
                <CarbonPanel carbon={data.carbon} />
              </div>
            </div>
          ) : phase === 'uploading' ? (
            <div className="result-panel is-pending" aria-busy="true">
              <span className="spinner" aria-hidden="true" />
              <p>正在推理，请稍候…</p>
            </div>
          ) : (
            <div className="result-panel is-empty">
              <span className="kicker">Result</span>
              <p>推理结果将显示在这里。</p>
            </div>
          )}
        </div>
      </div>

      {recent.length > 0 && (
        <section className="recent-panel" aria-labelledby="recent-heading">
          <h2 id="recent-heading" className="section-title">
            <span className="kicker">History</span>
            最近 5 次推理
          </h2>
          <ol className="recent-list">
            {recent.map((entry) => {
              const carbon = formatCarbon(entry.co2eqG)
              return (
                <li key={entry.id} className="recent-item">
                  <span className="recent-class">{classLabel(entry.className)}</span>
                  <span className="mono">{formatPercent(entry.confidence)}</span>
                  <span className="mono recent-carbon">
                    {carbon.text} {carbon.unit} CO₂eq
                  </span>
                  <span className="mono recent-model">{entry.model}</span>
                </li>
              )
            })}
          </ol>
        </section>
      )}
    </section>
  )
}
