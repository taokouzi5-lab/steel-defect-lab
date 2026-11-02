import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react'

import { formatBytes } from '../lib/format'
import { checkUploadFile, MAX_UPLOAD_BYTES } from '../lib/uploadCheck'
import type { UploadCheckResult } from '../lib/uploadCheck'

type Props = {
  onSelect: (file: File) => void
  onClear: () => void
  previewUrl: string | null
  isUploading: boolean
}

export function UploadZone({ onSelect, onClear, previewUrl, isUploading }: Props) {
  const [dragActive, setDragActive] = useState(false)
  const [rejection, setRejection] = useState<UploadCheckResult | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (file: File | null | undefined) => {
      if (!file || isUploading) return
      const result = checkUploadFile(file)
      if (!result.ok) {
        setRejection(result)
        return
      }
      setRejection(null)
      onSelect(file)
    },
    [isUploading, onSelect],
  )

  useEffect(() => {
    const handlePaste = (event: ClipboardEvent) => {
      const items = event.clipboardData?.items
      if (!items) return
      for (const item of Array.from(items)) {
        if (item.kind === 'file' && item.type.startsWith('image/')) {
          handleFile(item.getAsFile())
          break
        }
      }
    }
    window.addEventListener('paste', handlePaste)
    return () => window.removeEventListener('paste', handlePaste)
  }, [handleFile])

  const openPicker = useCallback(() => {
    if (!isUploading) inputRef.current?.click()
  }, [isUploading])

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
    handleFile(event.dataTransfer.files?.[0])
  }

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    handleFile(event.target.files?.[0])
    event.target.value = ''
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      openPicker()
    }
  }

  return (
    <div className="upload-area">
      <div
        className={`upload-zone${dragActive ? ' is-dragging' : ''}${isUploading ? ' is-uploading' : ''}`}
        role="button"
        tabIndex={0}
        aria-label="上传钢材表面图片"
        onClick={openPicker}
        onKeyDown={handleKeyDown}
        onDragOver={(event) => {
          event.preventDefault()
          if (!isUploading) setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        {isUploading ? (
          <div className="upload-state">
            <span className="spinner" aria-hidden="true" />
            <strong>正在上传并推理…</strong>
          </div>
        ) : previewUrl ? (
          <div className="upload-preview">
            <img src={previewUrl} alt="待推理的钢材表面图片" />
            <div className="upload-preview-actions">
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  openPicker()
                }}
              >
                重新选择
              </button>
              <button
                type="button"
                className="upload-clear"
                onClick={(event) => {
                  event.stopPropagation()
                  onClear()
                }}
              >
                移除
              </button>
            </div>
          </div>
        ) : (
          <div className="upload-state">
            <span className="upload-icon" aria-hidden="true">
              ▣
            </span>
            <strong>拖拽图片到这里</strong>
            <span className="upload-hint">或点击选择 · 支持直接粘贴（Ctrl+V）</span>
            <span className="upload-hint mono">
              JPEG / PNG · 最大 {formatBytes(MAX_UPLOAD_BYTES)}
            </span>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          className="upload-input"
          onChange={handleInputChange}
        />
      </div>
      {rejection !== null && !rejection.ok && (
        <p className="upload-rejection" role="alert">
          {rejection.message}
        </p>
      )}
    </div>
  )
}
