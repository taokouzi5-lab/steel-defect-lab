/** Response shapes of the DefectLens backend (mirrors app/schemas/predict.py). */

export interface HealthInfo {
  status: string
  models_loaded: string[]
}

export interface TopKItem {
  class_id: number
  class_name: string
  confidence: number
}

export interface GradCamData {
  base64: string
  format: string
}

export interface CarbonData {
  co2eq_g: number
  energy_kwh: number
  method: string
  equivalent: string
}

export interface PredictData {
  class_id: number
  class_name: string
  confidence: number
  top_k: TopKItem[]
  latency_ms: number
  gradcam_ms: number | null
  gradcam: GradCamData | null
  carbon: CarbonData
}

export interface ModelInfo {
  id: string
  arch: string
  quantization: string
  params: number
  test_acc: number
  test_macro_f1: number
  onnx_size_mb: number
  cpu_latency_ms: { p50: number; p95: number }
  training_carbon_co2eq_g: number
  inference_carbon_co2eq_g: number
  inference_carbon_method: string
  available: boolean
}

export interface SampleInfo {
  id: string
  filename: string
  label: string
}

export interface CarbonMeta {
  formula: string
  constants: Record<string, unknown>
  boundaries: string[]
  methods: string[]
}

export interface Envelope<T> {
  success: boolean
  data: T | null
  error: { code: string; message: string; status: number } | null
  meta: Record<string, unknown>
}
