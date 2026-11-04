import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'

import { VALID_JPEG, makeTempFile, runPredictWithValidImage } from './helpers'

interface FocusSnapshot {
  tag: string
  href: string | null
  aria: string | null
  text: string
}

const NAV_LABELS = ['推理', '示例库', '模型对比', '方法论']

const UPLOAD_ZONE_ARIA_LABEL = '上传钢材表面图片'

async function captureFocus(page: Page): Promise<FocusSnapshot | null> {
  return page.evaluate(() => {
    const el = document.activeElement
    if (!(el instanceof HTMLElement)) return null
    return {
      tag: el.tagName,
      href: el.getAttribute('href'),
      aria: el.getAttribute('aria-label'),
      text: (el.textContent ?? '').trim().slice(0, 24),
    }
  })
}

/**
 * 键盘可达性：
 * - Tab 可聚焦 4 个导航链接与上传按钮
 * - UploadZone 具备按钮语义（role/tabindex），Enter 键可触发文件选择
 * - 错误提示用 role="alert"，示例结果区用 aria-live="polite"
 * - 模型选择 radiogroup 与 Grad-CAM tablist 语义
 */
test.describe('键盘可达性与语义', () => {
  test('Tab 可依次聚焦 4 个导航链接与上传按钮', async ({ page }) => {
    await page.goto('/')
    // 等模型列表加载完（radio chips 也参与 Tab 序列）
    await expect(page.locator('.model-picker[role="radiogroup"]')).toBeVisible()

    const focusLog: FocusSnapshot[] = []
    for (let step = 0; step < 30; step += 1) {
      const snapshot = await captureFocus(page)
      if (snapshot) focusLog.push(snapshot)
      await page.keyboard.press('Tab')
    }

    for (const label of NAV_LABELS) {
      expect(
        focusLog.some((s) => s.text === label),
        `导航链接「${label}」应可通过 Tab 聚焦`,
      ).toBe(true)
    }
    expect(
      focusLog.some((s) => s.aria === UPLOAD_ZONE_ARIA_LABEL),
      '上传区域应可通过 Tab 聚焦',
    ).toBe(true)
  })

  test('UploadZone 具备按钮语义且 Enter 键可触发文件选择', async ({ page }) => {
    await page.goto('/')

    const zone = page.locator('.upload-zone')
    await expect(zone).toHaveAttribute('role', 'button')
    await expect(zone).toHaveAttribute('tabindex', '0')
    await expect(zone).toHaveAttribute('aria-label', UPLOAD_ZONE_ARIA_LABEL)

    await zone.focus()
    await expect(zone).toBeFocused()

    // Enter 触发隐藏 input 的 click，打开系统文件选择器（Playwright 拦截该事件）
    const [chooser] = await Promise.all([
      page.waitForEvent('filechooser'),
      zone.press('Enter'),
    ])
    await chooser.setFiles(VALID_JPEG)

    // 键盘选择成功：预览出现、预测按钮启用
    await expect(page.locator('.upload-preview img')).toBeVisible()
    await expect(page.locator('.predict-button')).toBeEnabled()
  })

  test('错误提示使用 role="alert"，示例结果区使用 aria-live="polite"', async ({ page }) => {
    await page.goto('/')

    // 客户端预检拒绝（.txt 扩展名，不发网络请求）
    const txtFile = makeTempFile('notes.txt', 'not an image')
    await page.locator('.upload-input').setInputFiles(txtFile)
    const rejection = page.locator('.upload-rejection[role="alert"]')
    await expect(rejection).toBeVisible()
    await expect(rejection).toContainText('格式')

    // 示例结果区带 aria-live，推理结果对屏幕阅读器播报
    await page.goto('/samples')
    await expect(page.locator('.sample-card')).toHaveCount(12)
    await expect(page.locator('.sample-result')).toHaveAttribute('aria-live', 'polite')
  })

  test('结果区语义：模型选择 radiogroup 与 Grad-CAM tablist', async ({ page }) => {
    await page.goto('/')

    const picker = page.locator('.model-picker[role="radiogroup"]')
    await expect(picker).toBeVisible()
    await expect(picker.locator('button[role="radio"]').first()).toBeVisible()

    await runPredictWithValidImage(page)

    const tablist = page.locator('.gradcam-tabs[role="tablist"]')
    await expect(tablist).toBeVisible()
    await expect(tablist.locator('[role="tab"]')).toHaveCount(3)
    await expect(tablist.locator('[role="tab"][aria-selected="true"]')).toHaveCount(1)

    // 碳成本面板带语义标签
    await expect(page.locator('.carbon-panel[aria-label="本次推理碳成本"]')).toBeVisible()
  })
})
