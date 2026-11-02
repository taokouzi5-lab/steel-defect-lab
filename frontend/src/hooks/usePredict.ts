/**
 * Inference request state machine (idle -> uploading -> done | error).
 *
 * Owns the AbortController: starting a new run aborts the previous in-flight
 * request, and unmounting aborts whatever is still pending, so a slow
 * inference can never set state on a dead component.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { AppError, NETWORK_ERROR_MESSAGE, predictSample, predictUpload } from '../api/client'
import type { PredictData } from '../api/types'

export type PredictPhase = 'idle' | 'uploading' | 'done' | 'error'

export type PredictRequest = { kind: 'file'; file: File } | { kind: 'sample'; sampleId: string }

export interface PredictOptions {
  model?: string
  gradcam?: boolean
}

export interface UsePredictResult {
  phase: PredictPhase
  data: PredictData | null
  error: AppError | null
  run: (request: PredictRequest, options?: PredictOptions) => Promise<PredictData | null>
}

export function usePredict(): UsePredictResult {
  const [phase, setPhase] = useState<PredictPhase>('idle')
  const [data, setData] = useState<PredictData | null>(null)
  const [error, setError] = useState<AppError | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  const run = useCallback(
    async (request: PredictRequest, options?: PredictOptions): Promise<PredictData | null> => {
      controllerRef.current?.abort()
      const controller = new AbortController()
      controllerRef.current = controller
      setPhase('uploading')
      setError(null)
      setData(null)
      try {
        const result =
          request.kind === 'file'
            ? await predictUpload(request.file, {
                model: options?.model,
                gradcam: options?.gradcam,
                signal: controller.signal,
              })
            : await predictSample(request.sampleId, {
                model: options?.model,
                gradcam: options?.gradcam,
                signal: controller.signal,
              })
        setData(result)
        setPhase('done')
        return result
      } catch (err) {
        if (controller.signal.aborted) return null
        const appError = err instanceof AppError ? err : new AppError('network', NETWORK_ERROR_MESSAGE)
        setError(appError)
        setPhase('error')
        return null
      }
    },
    [],
  )

  useEffect(() => {
    return () => {
      controllerRef.current?.abort()
    }
  }, [])

  return { phase, data, error, run }
}
