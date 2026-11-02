/**
 * Typed API client for the DefectLens backend.
 *
 * Every failure path — network, abort, HTTP, business error envelope,
 * unparseable body — is normalized into a single ``AppError`` with a
 * user-friendly Chinese message, so pages only ever handle one error shape.
 */

import { formatBytes } from '../lib/format'
import { MAX_UPLOAD_BYTES } from '../lib/uploadCheck'
import type {
  CarbonMeta,
  Envelope,
  HealthInfo,
  ModelInfo,
  PredictData,
  SampleInfo,
} from './types'

export const DEFAULT_MODEL_ID = 'resnet18-int8'

/** Same-origin by default (vite dev proxy / production origin); tests can point elsewhere. */
const API_BASE = import.meta.env.VITE_API_BASE ?? ''

const apiUrl = (path: string): string => `${API_BASE}${path}`

export type AppErrorKind = 'network' | 'aborted' | 'http' | 'business' | 'parse'

export class AppError extends Error {
  readonly kind: AppErrorKind
  readonly status: number | null
  readonly code: string | null

  constructor(
    kind: AppErrorKind,
    message: string,
    status: number | null = null,
    code: string | null = null,
  ) {
    super(message)
    this.name = 'AppError'
    this.kind = kind
    this.status = status
    this.code = code
  }
}

export const NETWORK_ERROR_MESSAGE = '网络连接失败，请检查后端服务是否已启动'

const FRIENDLY_ERROR_MESSAGES: Record<string, string> = {
  unsupported_media_type: '仅支持 JPEG/PNG 格式的图片',
  payload_too_large: `图片超过 ${formatBytes(MAX_UPLOAD_BYTES)} 大小上限，请压缩后再试`,
  unprocessable_image: '图片无法解析，文件可能已损坏或截断',
  validation_error: '请求参数无效',
  invalid_request: '请求参数无效',
  model_not_found: '未知的模型 ID',
  sample_not_found: '未知的示例 ID',
  model_not_ready: '模型暂未就绪，请稍后再试',
  inference_timeout: '推理超时，请稍后再试',
  rate_limited: '请求过于频繁，请稍后再试',
  internal_error: '服务内部错误，请稍后再试',
}

function friendlyBusinessMessage(code: string, serverMessage: string): string {
  return FRIENDLY_ERROR_MESSAGES[code] ?? (serverMessage || '请求失败')
}

async function parseJsonSafe(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch {
    return null
  }
}

function toFetchError(err: unknown): AppError {
  // jsdom's DOMException does not extend Error, so duck-type on the name.
  if (
    typeof err === 'object' &&
    err !== null &&
    'name' in err &&
    (err as { name: unknown }).name === 'AbortError'
  ) {
    return new AppError('aborted', '请求已取消')
  }
  return new AppError('network', NETWORK_ERROR_MESSAGE)
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch (err) {
    throw toFetchError(err)
  }

  const body = await parseJsonSafe(response)

  if (response.ok) {
    if (body !== null && typeof body === 'object' && 'success' in body) {
      const envelope = body as Envelope<T>
      if (envelope.success === true) return envelope.data as T
      const detail = envelope.error
      if (detail) {
        throw new AppError(
          'business',
          friendlyBusinessMessage(detail.code, detail.message),
          detail.status,
          detail.code,
        )
      }
    }
    throw new AppError('parse', '服务返回了无法解析的数据')
  }

  if (body !== null && typeof body === 'object' && 'error' in body) {
    const detail = (body as Envelope<T>).error
    if (detail) {
      throw new AppError(
        'business',
        friendlyBusinessMessage(detail.code, detail.message),
        detail.status,
        detail.code,
      )
    }
  }
  throw new AppError('http', `服务暂时不可用（HTTP ${response.status}）`, response.status)
}

export async function getHealth(signal?: AbortSignal): Promise<HealthInfo> {
  let response: Response
  try {
    response = await fetch(apiUrl('/api/v1/health'), { signal })
  } catch (err) {
    throw toFetchError(err)
  }
  if (!response.ok) {
    throw new AppError('http', `服务暂时不可用（HTTP ${response.status}）`, response.status)
  }
  const body = await parseJsonSafe(response)
  if (body === null || typeof body !== 'object') {
    throw new AppError('parse', '服务返回了无法解析的数据')
  }
  return body as HealthInfo
}

export function getModels(signal?: AbortSignal): Promise<ModelInfo[]> {
  return request<ModelInfo[]>(apiUrl('/api/v1/models'), { signal })
}

export function getSamples(signal?: AbortSignal): Promise<SampleInfo[]> {
  return request<SampleInfo[]>(apiUrl('/api/v1/samples'), { signal })
}

export function getCarbonMeta(signal?: AbortSignal): Promise<CarbonMeta> {
  return request<CarbonMeta>(apiUrl('/api/v1/meta'), { signal })
}

export interface PredictOptions {
  model?: string
  gradcam?: boolean
  signal?: AbortSignal
}

export function predictUpload(file: File, options: PredictOptions = {}): Promise<PredictData> {
  const body = new FormData()
  body.append('file', file)
  body.append('model', options.model ?? DEFAULT_MODEL_ID)
  body.append('gradcam', String(options.gradcam ?? false))
  return request<PredictData>(apiUrl('/api/v1/predict'), {
    method: 'POST',
    body,
    signal: options.signal,
  })
}

export function predictSample(
  sampleId: string,
  options: PredictOptions = {},
): Promise<PredictData> {
  const body = new FormData()
  body.append('model', options.model ?? DEFAULT_MODEL_ID)
  body.append('gradcam', String(options.gradcam ?? false))
  return request<PredictData>(
    apiUrl(`/api/v1/predict/sample/${encodeURIComponent(sampleId)}`),
    {
      method: 'POST',
      body,
      signal: options.signal,
    },
  )
}
