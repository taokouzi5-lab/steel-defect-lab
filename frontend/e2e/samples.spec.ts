import { expect, test } from '@playwright/test'

import { parsePercentText, resultPanel } from './helpers'

/**
 * 示例推演流程：示例库 12 张卡片（6 类 × 2 张），点击第一张走
 * upload-free 通道，结果面板出现且置信度数值合法。
 */
test.describe('示例推演', () => {
  test('示例库加载 12 张卡片，点击第一张推演出结果面板', async ({ page }) => {
    await page.goto('/samples')

    const cards = page.locator('.sample-card')
    await expect(cards).toHaveCount(12)

    const firstCard = cards.first()
    await firstCard.click()

    const panel = resultPanel(page.locator('.sample-result'))
    await expect(panel).toBeVisible()

    // 分类名非空
    const heading = panel.locator('.result-head h2')
    await expect(heading).not.toBeEmpty()

    // 置信度显示且数值落在 0-100%
    const confidence = panel.locator('.result-confidence.mono')
    await expect(confidence).toBeVisible()
    await expect(confidence).not.toBeEmpty()
    const pct = parsePercentText(await confidence.textContent())
    expect(Number.isFinite(pct)).toBe(true)
    expect(pct).toBeGreaterThanOrEqual(0)
    expect(pct).toBeLessThanOrEqual(100)

    // 推理延迟非空
    await expect(panel.locator('.result-latency.mono')).toContainText('推理')

    // 碳成本面板同步出现
    await expect(panel.locator('.carbon-panel')).toBeVisible()

    // 当前卡片高亮
    await expect(firstCard).toHaveClass(/is-active/)
  })
})
