import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { AppError, predictSample, predictUpload } from '../api/client'
import type { PredictData } from '../api/types'
import { usePredict } from './usePredict'

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return { ...actual, predictUpload: vi.fn(), predictSample: vi.fn() }
})

const payload: PredictData = {
  class_id: 5,
  class_name: 'scratches',
  confidence: 0.9876,
  top_k: [
    { class_id: 5, class_name: 'scratches', confidence: 0.9876 },
    { class_id: 0, class_name: 'crazing', confidence: 0.011 },
    { class_id: 1, class_name: 'inclusion', confidence: 0.0012 },
  ],
  latency_ms: 12.4,
  gradcam_ms: null,
  gradcam: null,
  carbon: {
    co2eq_g: 7.4e-5,
    energy_kwh: 4.57e-8,
    method: 'tier-representative-fallback',
    equivalent: '≈ 手机充电 0.00074 秒',
  },
}

const file = () => new File(['x'], 'a.jpg', { type: 'image/jpeg' })

beforeEach(() => {
  vi.mocked(predictUpload).mockReset()
  vi.mocked(predictSample).mockReset()
})

describe('usePredict', () => {
  it('starts idle with no data and no error', () => {
    const { result } = renderHook(() => usePredict())
    expect(result.current.phase).toBe('idle')
    expect(result.current.data).toBeNull()
    expect(result.current.error).toBeNull()
  })

  it('walks idle -> uploading -> done for a successful file predict', async () => {
    let resolveRun!: (data: typeof payload) => void
    vi.mocked(predictUpload).mockImplementation(
      () => new Promise((resolve) => { resolveRun = resolve }),
    )

    const { result } = renderHook(() => usePredict())
    let pending: Promise<PredictData | null> | undefined
    act(() => {
      pending = result.current.run({ kind: 'file', file: file() })
    })
    expect(result.current.phase).toBe('uploading')
    expect(result.current.data).toBeNull()

    await act(async () => {
      resolveRun(payload)
      await pending
    })
    expect(result.current.phase).toBe('done')
    expect(result.current.data).toEqual(payload)
    expect(result.current.error).toBeNull()
  })

  it('walks idle -> uploading -> error when the request fails', async () => {
    vi.mocked(predictUpload).mockRejectedValue(
      new AppError('business', '模型暂未就绪，请稍后再试', 503, 'model_not_ready'),
    )
    const { result } = renderHook(() => usePredict())
    await act(async () => {
      await result.current.run({ kind: 'file', file: file() })
    })
    expect(result.current.phase).toBe('error')
    expect(result.current.error?.kind).toBe('business')
    expect(result.current.error?.message).toBe('模型暂未就绪，请稍后再试')
    expect(result.current.data).toBeNull()
  })

  it('normalizes unexpected rejections into a network AppError', async () => {
    vi.mocked(predictUpload).mockRejectedValue(new Error('boom'))
    const { result } = renderHook(() => usePredict())
    await act(async () => {
      await result.current.run({ kind: 'file', file: file() })
    })
    expect(result.current.phase).toBe('error')
    expect(result.current.error?.kind).toBe('network')
  })

  it('routes sample requests to predictSample with the sample id', async () => {
    vi.mocked(predictSample).mockResolvedValue(payload)
    const { result } = renderHook(() => usePredict())
    await act(async () => {
      await result.current.run({ kind: 'sample', sampleId: 'crazing_101' }, { gradcam: true })
    })
    expect(vi.mocked(predictSample)).toHaveBeenCalledWith('crazing_101', {
      model: undefined,
      gradcam: true,
      signal: expect.any(AbortSignal),
    })
    expect(result.current.phase).toBe('done')
    expect(result.current.data).toEqual(payload)
  })

  it('aborts the in-flight request when the component unmounts', async () => {
    let capturedSignal: AbortSignal | undefined
    vi.mocked(predictUpload).mockImplementation((_file, options) => {
      capturedSignal = options?.signal
      return new Promise((_resolve, reject) => {
        capturedSignal?.addEventListener('abort', () =>
          reject(new DOMException('Aborted', 'AbortError')),
        )
      })
    })
    const { result, unmount } = renderHook(() => usePredict())
    act(() => {
      void result.current.run({ kind: 'file', file: file() })
    })
    expect(capturedSignal?.aborted).toBe(false)
    unmount()
    expect(capturedSignal?.aborted).toBe(true)
  })

  it('aborts the previous request when a new run starts', async () => {
    const signals: AbortSignal[] = []
    vi.mocked(predictUpload).mockImplementation((_file, options) => {
      const signal = options?.signal
      signals.push(signal as AbortSignal)
      return new Promise((_resolve, reject) => {
        signal?.addEventListener('abort', () =>
          reject(new DOMException('Aborted', 'AbortError')),
        )
      })
    })
    const { result } = renderHook(() => usePredict())
    act(() => {
      void result.current.run({ kind: 'file', file: file() })
    })
    expect(signals[0].aborted).toBe(false)

    vi.mocked(predictUpload).mockResolvedValue(payload)
    await act(async () => {
      await result.current.run({ kind: 'file', file: file() })
    })
    expect(signals[0].aborted).toBe(true)
    expect(result.current.phase).toBe('done')
    expect(result.current.data).toEqual(payload)
  })
})
