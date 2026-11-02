import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  AppError,
  getCarbonMeta,
  getHealth,
  getModels,
  getSamples,
  predictSample,
  predictUpload,
} from './client'

const fetchMock = vi.fn()

function jsonBody(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function errorEnvelope(code: string, message: string, status: number): unknown {
  return {
    success: false,
    data: null,
    error: { code, message, status },
    meta: { request_id: 'test-request' },
  }
}

const predictPayload = {
  class_id: 5,
  class_name: 'scratches',
  confidence: 0.9876,
  top_k: [
    { class_id: 5, class_name: 'scratches', confidence: 0.9876 },
    { class_id: 0, class_name: 'crazing', confidence: 0.011 },
    { class_id: 1, class_name: 'inclusion', confidence: 0.0012 },
  ],
  latency_ms: 12.4,
  gradcam_ms: 45.2,
  gradcam: { base64: 'AAAA', format: 'png' },
  carbon: {
    co2eq_g: 7.4e-5,
    energy_kwh: 4.57e-8,
    method: 'tier-representative-fallback',
    equivalent: '≈ 手机充电 0.00074 秒',
  },
}

function fileFixture(): File {
  return new File(['fake-bytes'], 'steel.jpg', { type: 'image/jpeg' })
}

function successEnvelope(data: unknown): unknown {
  return { success: true, data, error: null, meta: { request_id: 'test-request' } }
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('getHealth', () => {
  it('parses the plain (non-envelope) health response', async () => {
    fetchMock.mockResolvedValueOnce(jsonBody({ status: 'ok', models_loaded: ['resnet18-int8'] }))
    await expect(getHealth()).resolves.toEqual({ status: 'ok', models_loaded: ['resnet18-int8'] })
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/health', expect.anything())
  })
})

describe('getModels', () => {
  it('unwraps the success envelope and returns the model list', async () => {
    const models = [{ id: 'resnet18-int8' }]
    fetchMock.mockResolvedValueOnce(jsonBody(successEnvelope(models)))
    await expect(getModels()).resolves.toEqual(models)
  })
})

describe('getSamples', () => {
  it('unwraps the sample list', async () => {
    const samples = [
      { id: 'crazing_101', filename: 'crazing_101.jpg', label: 'crazing' },
      { id: 'crazing_11', filename: 'crazing_11.jpg', label: 'crazing' },
    ]
    fetchMock.mockResolvedValueOnce(jsonBody(successEnvelope(samples)))
    await expect(getSamples()).resolves.toEqual(samples)
  })
})

describe('getCarbonMeta', () => {
  it('unwraps the methodology disclosure', async () => {
    const meta = {
      formula: 'co2eq_g = energy_kwh_per_inference * grid_carbon_intensity_g_per_kwh',
      constants: { country: 'China', pue: 2 },
      boundaries: ['covers inference electricity only'],
      methods: ['mace-f-methodD-measured'],
    }
    fetchMock.mockResolvedValueOnce(jsonBody(successEnvelope(meta)))
    await expect(getCarbonMeta()).resolves.toEqual(meta)
  })
})

describe('predictUpload', () => {
  it('posts multipart form data with file, model and gradcam fields', async () => {
    fetchMock.mockResolvedValueOnce(jsonBody(successEnvelope(predictPayload)))
    const file = fileFixture()
    const result = await predictUpload(file, { model: 'resnet18-int8', gradcam: true })
    expect(result).toEqual(predictPayload)

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/v1/predict')
    expect(init?.method).toBe('POST')
    const body = init?.body as FormData
    expect(body.get('file')).toBe(file)
    expect(body.get('model')).toBe('resnet18-int8')
    expect(body.get('gradcam')).toBe('true')
  })

  it('defaults model to resnet18-int8 and gradcam to false', async () => {
    fetchMock.mockResolvedValueOnce(jsonBody(successEnvelope(predictPayload)))
    await predictUpload(fileFixture())
    const body = fetchMock.mock.calls[0][1]?.body as FormData
    expect(body.get('model')).toBe('resnet18-int8')
    expect(body.get('gradcam')).toBe('false')
  })
})

describe('predictSample', () => {
  it('posts to the sample endpoint with the model and gradcam fields', async () => {
    fetchMock.mockResolvedValueOnce(jsonBody(successEnvelope(predictPayload)))
    const result = await predictSample('crazing_101', { model: 'resnet50-fp32', gradcam: true })
    expect(result).toEqual(predictPayload)

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/v1/predict/sample/crazing_101')
    const body = init?.body as FormData
    expect(body.get('model')).toBe('resnet50-fp32')
    expect(body.get('gradcam')).toBe('true')
    expect(body.get('file')).toBeNull()
  })
})

describe('business errors', () => {
  it.each([
    [415, 'unsupported_media_type', 'only JPEG and PNG images are accepted', '仅支持 JPEG/PNG 格式的图片'],
    [413, 'payload_too_large', 'upload exceeds the maximum allowed size', '图片超过 5.0 MB 大小上限，请压缩后再试'],
    [422, 'unprocessable_image', 'image could not be decoded', '图片无法解析，文件可能已损坏或截断'],
    [429, 'rate_limited', 'request rate limit exceeded', '请求过于频繁，请稍后再试'],
    [503, 'model_not_ready', 'the requested model is not available yet', '模型暂未就绪，请稍后再试'],
  ])(
    'normalizes HTTP %s %s into a friendly business AppError',
    async (status, code, serverMessage, friendly) => {
      fetchMock.mockResolvedValueOnce(jsonBody(errorEnvelope(code, serverMessage, status), status))
      const err = await predictUpload(fileFixture()).catch((e: unknown) => e)
      expect(err).toBeInstanceOf(AppError)
      const appErr = err as AppError
      expect(appErr.kind).toBe('business')
      expect(appErr.status).toBe(status)
      expect(appErr.code).toBe(code)
      expect(appErr.message).toBe(friendly)
    },
  )
})

describe('transport errors', () => {
  it('maps fetch network failures to a network AppError', async () => {
    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'))
    const err = await getModels().catch((e: unknown) => e)
    expect(err).toBeInstanceOf(AppError)
    const appErr = err as AppError
    expect(appErr.kind).toBe('network')
    expect(appErr.status).toBeNull()
    expect(appErr.code).toBeNull()
    expect(appErr.message).toBe('网络连接失败，请检查后端服务是否已启动')
  })

  it('maps aborted requests to an aborted AppError', async () => {
    fetchMock.mockRejectedValueOnce(new DOMException('The operation was aborted.', 'AbortError'))
    const err = await predictUpload(fileFixture()).catch((e: unknown) => e)
    expect((err as AppError).kind).toBe('aborted')
  })

  it('maps non-JSON HTTP errors (proxy 502) to an http AppError with the status', async () => {
    fetchMock.mockResolvedValueOnce(new Response('<html>Bad Gateway</html>', { status: 502 }))
    const err = await getSamples().catch((e: unknown) => e)
    const appErr = err as AppError
    expect(appErr.kind).toBe('http')
    expect(appErr.status).toBe(502)
    expect(appErr.message).toBe('服务暂时不可用（HTTP 502）')
  })

  it('maps a 200 response with an unparseable body to a parse AppError', async () => {
    fetchMock.mockResolvedValueOnce(new Response('not json at all', { status: 200 }))
    const err = await getCarbonMeta().catch((e: unknown) => e)
    expect((err as AppError).kind).toBe('parse')
  })

  it('treats success:false inside a 200 envelope as a business error', async () => {
    fetchMock.mockResolvedValueOnce(jsonBody(errorEnvelope('internal_error', 'boom', 500), 200))
    const err = await getModels().catch((e: unknown) => e)
    const appErr = err as AppError
    expect(appErr.kind).toBe('business')
    expect(appErr.code).toBe('internal_error')
  })
})
