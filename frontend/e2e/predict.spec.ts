import { expect, test } from '@playwright/test'

import {
  parseLatencyMs,
  parsePercentText,
  resultPanel,
  runPredictWithValidImage,
} from './helpers'

/**
 * 上传预测主流程。
 *
 * 确定性断言原则：不硬编码具体类别（模型资产更新后输出可能变化），
 * 只断言结构与数值范围（置信度 0-100%、推理延迟 > 0）。
 */
test.describe('上传预测主流程', () => {
  test('上传合法钢材图 → 开始推理 → 结果面板展示分类/置信度/碳成本/Grad-CAM', async ({
    page,
  }) => {
    await page.goto('/')

    const panel = await runPredictWithValidImage(page)

    // 分类名：非空即可（不硬编码具体类别）
    const heading = panel.locator('.result-head h2')
    await expect(heading).toBeVisible()
    await expect(heading).not.toBeEmpty()

    // 置信度：mono 元素非空，数值落在 0-100%
    const confidence = panel.locator('.result-confidence.mono')
    await expect(confidence).toBeVisible()
    await expect(confidence).not.toBeEmpty()
    const confidenceText = await confidence.textContent()
    const pct = parsePercentText(confidenceText)
    expect(
      Number.isFinite(pct),
      `置信度应可解析为数值，实际文本: ${confidenceText}`,
    ).toBe(true)
    expect(pct).toBeGreaterThanOrEqual(0)
    expect(pct).toBeLessThanOrEqual(100)

    // 推理延迟：格式「推理 X ms/s」且 > 0
    const latency = panel.locator('.result-latency.mono')
    await expect(latency).toBeVisible()
    await expect(latency).toContainText('推理')
    const latencyText = await latency.textContent()
    const latencyMs = parseLatencyMs(latencyText)
    expect(latencyMs, `推理延迟应 > 0，实际文本: ${latencyText}`).not.toBeNull()
    expect(latencyMs ?? 0).toBeGreaterThan(0)

    // 置信度分布条（top-k 结构）
    await expect(panel.locator('.confidence-bars .confidence-row').first()).toBeVisible()

    // 碳成本面板：出现 gCO₂ 字样
    const carbon = panel.locator('.carbon-panel')
    await expect(carbon).toBeVisible()
    await expect(carbon).toContainText('CO₂eq')

    // Grad-CAM 区域出现带 alt 的图片
    await expect(panel.locator('.gradcam-viewer img[alt]').first()).toBeVisible()

    // 最近推理历史新增一条记录
    await expect(page.locator('.recent-panel .recent-item')).toHaveCount(1)
  })

  test('空态时不显示结果面板，预测按钮禁用', async ({ page }) => {
    await page.goto('/')

    await expect(page.locator('.result-panel.is-empty')).toBeVisible()
    await expect(resultPanel(page)).toHaveCount(0)
    await expect(page.locator('.predict-button')).toBeDisabled()
  })
})
