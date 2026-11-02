import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import Home from './Home'

// Fixtures captured from the live backend (GET /models, POST /predict).
const modelsPayload = {
  success: true,
  data: [
    {
      id: 'resnet18-fp32',
      arch: 'resnet18',
      quantization: 'fp32',
      params: 11179590,
      test_acc: 0.9944,
      test_macro_f1: 0.9944,
      onnx_size_mb: 42.6377,
      cpu_latency_ms: { p50: 9.317, p95: 10.322 },
      training_carbon_co2eq_g: 0.3085,
      inference_carbon_co2eq_g: 7.4018e-5,
      inference_carbon_method: 'tier-representative-fallback',
      available: true,
    },
    {
      id: 'resnet18-int8',
      arch: 'resnet18',
      quantization: 'int8',
      params: 11179590,
      test_acc: 0.9944,
      test_macro_f1: 0.9944,
      onnx_size_mb: 10.7735,
      cpu_latency_ms: { p50: 10.141, p95: 11.449 },
      training_carbon_co2eq_g: 0.3085,
      inference_carbon_co2eq_g: 7.4018e-5,
      inference_carbon_method: 'tier-representative-fallback',
      available: true,
    },
    {
      id: 'resnet50-fp32',
      arch: 'resnet50',
      quantization: 'fp32',
      params: 23520326,
      test_acc: 0.9889,
      test_macro_f1: 0.9889,
      onnx_size_mb: 89.6449,
      cpu_latency_ms: { p50: 21.903, p95: 25.098 },
      training_carbon_co2eq_g: 0.4511,
      inference_carbon_co2eq_g: 7.4018e-5,
      inference_carbon_method: 'mace-f-methodD-measured',
      available: true,
    },
    {
      id: 'resnet50-int8',
      arch: 'resnet50',
      quantization: 'int8',
      params: 23520326,
      test_acc: 0.9889,
      test_macro_f1: 0.9889,
      onnx_size_mb: 22.9146,
      cpu_latency_ms: { p50: 20.346, p95: 21.729 },
      training_carbon_co2eq_g: 0.4511,
      inference_carbon_co2eq_g: 7.4018e-5,
      inference_carbon_method: 'mace-f-methodD-measured',
      available: true,
    },
  ],
  error: null,
  meta: { count: 4 },
}

const predictPayload = {
  success: true,
  data: {
    class_id: 1,
    class_name: 'inclusion',
    confidence: 0.4769796677874598,
    top_k: [
      { class_id: 1, class_name: 'inclusion', confidence: 0.4769796677874598 },
      { class_id: 5, class_name: 'scratches', confidence: 0.4174403301557142 },
      { class_id: 3, class_name: 'pitted_surface', confidence: 0.09630151906255885 },
    ],
    latency_ms: 10.746100000687875,
    gradcam_ms: 81.39300000038929,
    gradcam: { base64: 'AAAA', format: 'png' },
    carbon: {
      co2eq_g: 7.401844208667446e-5,
      energy_kwh: 4.574800494862325e-8,
      method: 'tier-representative-fallback',
      equivalent: '≈ 手机充电 0.00074 秒',
    },
  },
  error: null,
  meta: { model: 'resnet18-int8', request_id: 'test-request' },
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

const fetchMock = vi.fn()

beforeEach(() => {
  fetchMock.mockReset()
  fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url === '/api/v1/models') return jsonResponse(modelsPayload)
    if (url === '/api/v1/predict') return jsonResponse(predictPayload)
    throw new Error(`unmocked fetch: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('Home page (render smoke)', () => {
  it('loads models, runs a dropped file through predict and renders the result data', async () => {
    render(
      <MemoryRouter>
        <Home />
      </MemoryRouter>,
    )

    // Model picker filled from /models.
    expect(await screen.findAllByRole('radio')).toHaveLength(4)
    expect(screen.getByRole('radio', { name: /resnet18 int8/ })).toHaveAttribute(
      'aria-checked',
      'true',
    )

    // Upload zone accepts a dropped JPEG.
    const zone = screen.getByRole('button', { name: '上传钢材表面图片' })
    const file = new File(['\xff\xd8\xffsteel'], 'steel.jpg', { type: 'image/jpeg' })
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })

    const predictButton = screen.getByRole('button', { name: '开始推理' })
    expect(predictButton).toBeEnabled()
    fireEvent.click(predictButton)

    // Result panel renders the backend data through the components.
    expect((await screen.findAllByText('夹杂 Inclusion')).length).toBeGreaterThan(0)
    expect(screen.getAllByText('47.7%').length).toBeGreaterThan(0)
    expect(screen.getAllByText('划痕 Scratches').length).toBeGreaterThan(0)
    expect(screen.getByText('≈ 手机充电 0.00074 秒')).toBeTruthy()
    expect(screen.getAllByText('74.0').length).toBeGreaterThan(0)
    expect(screen.getAllByText(/µg CO₂eq/).length).toBeGreaterThan(0)
    expect(screen.getByText('推理 10.7 ms · Grad-CAM 81.4 ms')).toBeTruthy()
    expect(screen.getByText('T3 层级代表值回退')).toBeTruthy()

    // Grad-CAM viewer offers the three-state tabs with overlay active.
    expect(screen.getByRole('tab', { name: '叠加' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByAltText('Grad-CAM 热力图叠加层')).toBeTruthy()

    // Recent history records the run.
    expect(await screen.findByText('最近 5 次推理')).toBeTruthy()
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/predict',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('rejects a non-image drop with a friendly hint before any network call', async () => {
    render(
      <MemoryRouter>
        <Home />
      </MemoryRouter>,
    )

    await screen.findAllByRole('radio')
    const zone = screen.getByRole('button', { name: '上传钢材表面图片' })
    const textFile = new File(['notes'], 'notes.txt', { type: 'text/plain' })
    fireEvent.drop(zone, { dataTransfer: { files: [textFile] } })

    expect(screen.getByRole('alert')).toHaveTextContent('仅支持 JPEG/PNG 格式的图片')
    expect(screen.getByRole('button', { name: '开始推理' })).toBeDisabled()

    const predictCalls = fetchMock.mock.calls.filter(([url]) => String(url) === '/api/v1/predict')
    await waitFor(() => expect(predictCalls).toHaveLength(0))
  })
})
