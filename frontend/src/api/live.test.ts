// @vitest-environment node
// Node environment: native (undici) File/FormData serialize multipart correctly,
// which jsdom's implementations do not when handed to Node's fetch.
import { readFileSync } from 'node:fs'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import {
  getCarbonMeta,
  getHealth,
  getModels,
  getSamples,
  predictSample,
  predictUpload,
} from './client'

// Only runs against a real local backend: LIVE_API=1 npx vitest run src/api/live.test.ts
const LIVE = process.env.LIVE_API === '1'

const fixturePath = path.resolve(
  process.cwd(),
  '..',
  'backend',
  'tests',
  'fixtures',
  'valid_jpeg.jpg',
)

const CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

describe.skipIf(!LIVE)('live backend integration (LIVE_API=1)', () => {
  it('reports health with models loaded', async () => {
    const health = await getHealth()
    expect(health.status).toBe('ok')
    expect(health.models_loaded).toContain('resnet18-int8')
  })

  it('lists the four models with benchmark fields', async () => {
    const models = await getModels()
    expect(models).toHaveLength(4)
    for (const model of models) {
      expect(model.test_acc).toBeGreaterThan(0.9)
      expect(model.cpu_latency_ms.p50).toBeGreaterThan(0)
      expect(model.cpu_latency_ms.p95).toBeGreaterThan(0)
      expect(model.inference_carbon_co2eq_g).toBeGreaterThan(0)
      expect(model.training_carbon_co2eq_g).toBeGreaterThan(0)
    }
  })

  it('lists 12 demo samples', async () => {
    const samples = await getSamples()
    expect(samples).toHaveLength(12)
  })

  it('serves the carbon methodology disclosure', async () => {
    const meta = await getCarbonMeta()
    expect(meta.formula).toContain('co2eq_g')
    expect(meta.constants.country).toBe('China')
    expect(meta.boundaries.length).toBeGreaterThan(0)
    expect(meta.methods.length).toBeGreaterThan(0)
  })

  it('predicts on a real JPEG upload with Grad-CAM and carbon data', async () => {
    const bytes = new Uint8Array(readFileSync(fixturePath))
    const file = new File([bytes], 'valid_jpeg.jpg', { type: 'image/jpeg' })
    const data = await predictUpload(file, { model: 'resnet18-int8', gradcam: true })

    expect(CLASS_NAMES).toContain(data.class_name)
    expect(data.confidence).toBeGreaterThan(0)
    expect(data.confidence).toBeLessThanOrEqual(1)
    expect(data.top_k).toHaveLength(3)
    expect(data.latency_ms).toBeGreaterThan(0)
    expect(data.gradcam).not.toBeNull()
    if (data.gradcam) {
      expect(data.gradcam.format).toBe('png')
      expect(data.gradcam.base64.length).toBeGreaterThan(1000)
      expect(data.gradcam_ms).toBeGreaterThan(0)
    }
    expect(data.carbon.co2eq_g).toBeGreaterThan(0)
    expect(data.carbon.equivalent.length).toBeGreaterThan(0)
  })

  it('predicts on a demo sample through the upload-free channel', async () => {
    const data = await predictSample('crazing_101', { model: 'resnet18-int8', gradcam: false })
    expect(CLASS_NAMES).toContain(data.class_name)
    expect(data.top_k).toHaveLength(3)
    expect(data.gradcam).toBeNull()
    expect(data.carbon.co2eq_g).toBeGreaterThan(0)
  })
})
