import type { AppError } from '../api/client'
import type { ModelInfo } from '../api/types'

type Props = {
  models: ModelInfo[]
  selectedId: string
  isLoading: boolean
  error: AppError | null
  onSelect: (id: string) => void
  onRetry: () => void
}

export function ModelPicker({ models, selectedId, isLoading, error, onSelect, onRetry }: Props) {
  if (isLoading) {
    return (
      <div className="model-picker" aria-busy="true">
        <span className="mono">正在加载模型列表…</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="model-picker is-error" role="alert">
        <span>{error.message}</span>
        <button type="button" className="retry-button" onClick={onRetry}>
          重试
        </button>
      </div>
    )
  }

  return (
    <div className="model-picker" role="radiogroup" aria-label="选择模型">
      {models.map((model) => {
        const isSelected = model.id === selectedId
        return (
          <button
            key={model.id}
            type="button"
            role="radio"
            aria-checked={isSelected}
            aria-label={`${model.arch} ${model.quantization}`}
            className={`model-chip${isSelected ? ' is-selected' : ''}`}
            disabled={!model.available}
            onClick={() => onSelect(model.id)}
          >
            <span className="mono">{model.arch}</span>
            <span className="model-chip-tag">{model.quantization}</span>
          </button>
        )
      })}
    </div>
  )
}
