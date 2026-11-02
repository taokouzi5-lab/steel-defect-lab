import { useEffect, useState } from 'react'

import { AppError, NETWORK_ERROR_MESSAGE, getSamples } from '../api/client'
import type { SampleInfo } from '../api/types'
import { CarbonPanel } from '../components/CarbonPanel'
import { ConfidenceBars } from '../components/ConfidenceBars'
import { usePredict } from '../hooks/usePredict'
import { classLabel, formatMs, formatPercent } from '../lib/format'

type SampleGroup = { label: string; samples: SampleInfo[] }

function groupSamples(samples: SampleInfo[]): SampleGroup[] {
  const groups: SampleGroup[] = []
  const seen = new Map<string, SampleGroup>()
  for (const sample of samples) {
    const existing = seen.get(sample.label)
    if (existing) {
      existing.samples.push(sample)
    } else {
      const group: SampleGroup = { label: sample.label, samples: [sample] }
      seen.set(sample.label, group)
      groups.push(group)
    }
  }
  return groups
}

export default function Samples() {
  const [samples, setSamples] = useState<SampleInfo[]>([])
  const [loadError, setLoadError] = useState<AppError | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [activeId, setActiveId] = useState<string | null>(null)
  const { phase, data, error: predictError, run } = usePredict()

  useEffect(() => {
    let cancelled = false
    getSamples()
      .then((list) => {
        if (!cancelled) setSamples(list)
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setLoadError(
            err instanceof AppError ? err : new AppError('network', NETWORK_ERROR_MESSAGE),
          )
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const handleSampleClick = (sampleId: string) => {
    if (phase === 'uploading') return
    setActiveId(sampleId)
    void run({ kind: 'sample', sampleId })
  }

  const groups = groupSamples(samples)
  const showResult = phase === 'done' && data !== null

  return (
    <section aria-labelledby="samples-heading">
      <h1 id="samples-heading" className="section-title">
        <span className="kicker">Test Set</span>
        示例库
      </h1>
      <p className="page-lede">
        NEU-DET 测试集样本（未参与训练），6 类缺陷 × 2 张。点击任意样本直接推演，走
        upload-free 通道。
      </p>

      {isLoading && (
        <p className="page-status" aria-busy="true">
          正在加载示例…
        </p>
      )}
      {loadError !== null && (
        <p className="page-error" role="alert">
          {loadError.message}
        </p>
      )}

      <div className="samples-layout">
        <div className="samples-grid">
          {groups.map((group) => (
            <section key={group.label} className="sample-group" aria-label={classLabel(group.label)}>
              <h2 className="sample-group-title">{classLabel(group.label)}</h2>
              <div className="sample-cards">
                {group.samples.map((sample) => {
                  const isActive = sample.id === activeId
                  const isBusy = isActive && phase === 'uploading'
                  return (
                    <button
                      key={sample.id}
                      type="button"
                      className={`sample-card${isActive ? ' is-active' : ''}`}
                      disabled={phase === 'uploading'}
                      onClick={() => handleSampleClick(sample.id)}
                    >
                      <span className="sample-thumb" aria-hidden="true">
                        <span className="sample-thumb-index mono">{sample.id}</span>
                      </span>
                      <span className="sample-card-body">
                        <span className="mono">{sample.filename}</span>
                        <span className="sample-card-label">{classLabel(sample.label)}</span>
                      </span>
                      {isBusy && <span className="sample-card-busy" aria-hidden="true" />}
                    </button>
                  )
                })}
              </div>
            </section>
          ))}
        </div>

        <aside className="sample-result" aria-live="polite">
          {showResult && data ? (
            <div className="result-panel">
              <header className="result-head">
                <h2>{classLabel(data.class_name)}</h2>
                <span className="result-confidence mono">{formatPercent(data.confidence)}</span>
                <span className="result-latency mono">推理 {formatMs(data.latency_ms)}</span>
              </header>
              <ConfidenceBars items={data.top_k} highlightClass={data.class_name} />
              <CarbonPanel carbon={data.carbon} />
            </div>
          ) : phase === 'uploading' ? (
            <div className="result-panel is-pending" aria-busy="true">
              <span className="spinner" aria-hidden="true" />
              <p>正在推演示例…</p>
            </div>
          ) : (
            <div className="result-panel is-empty">
              <span className="kicker">Result</span>
              <p>点击左侧样本查看推演结果。</p>
            </div>
          )}
          {phase === 'error' && predictError !== null && (
            <p className="predict-error" role="alert">
              {predictError.message}
            </p>
          )}
        </aside>
      </div>
    </section>
  )
}
