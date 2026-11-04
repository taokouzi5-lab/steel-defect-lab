import path from 'node:path'

import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'

import {
  SCREENSHOTS_DIR,
  runPredictWithValidImage,
  waitForStableLayout,
} from './helpers'

interface Viewport {
  name: string
  width: number
  height: number
}

const VIEWPORTS: Viewport[] = [
  { name: '320', width: 320, height: 640 },
  { name: '768', width: 768, height: 1024 },
  { name: '1440', width: 1440, height: 900 },
]

/**
 * 依次在三个断点 resize、等布局稳定、截图，并收集横向溢出违规。
 * 截图全部产出后统一断言（允许 1px 亚像素取整误差），
 * 保证基线图在断言失败时也已落盘，违规点一次性全部列出。
 */
async function shootAtViewports(page: Page, pageName: 'home' | 'samples'): Promise<void> {
  const overflows: string[] = []
  for (const viewport of VIEWPORTS) {
    await page.setViewportSize({ width: viewport.width, height: viewport.height })
    await waitForStableLayout(page)

    const { scrollWidth, innerWidth } = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
    }))
    if (scrollWidth > innerWidth + 1) {
      overflows.push(`${pageName}@${viewport.name}: scrollWidth ${scrollWidth} > innerWidth ${innerWidth}`)
    }

    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, `${pageName}-${viewport.name}.png`),
      fullPage: true,
    })
  }
  expect(overflows, `存在横向溢出（scrollWidth > innerWidth + 1）:\n${overflows.join('\n')}`).toEqual(
    [],
  )
}

/**
 * 响应式截图回归：320/768/1440 三个断点。
 * 首页带一次真实预测结果（结果面板可见后再截图）。
 * 截图入库作为视觉回归基线（更新方式见 e2e/README.md）。
 */
test.describe('响应式截图回归', () => {
  test('首页（含预测结果）在 320/768/1440 断点无横向溢出并出图', async ({ page }) => {
    await page.goto('/')
    await runPredictWithValidImage(page)
    await shootAtViewports(page, 'home')
  })

  test('示例库在 320/768/1440 断点无横向溢出并出图', async ({ page }) => {
    await page.goto('/samples')
    await expect(page.locator('.sample-card')).toHaveCount(12)
    await shootAtViewports(page, 'samples')
  })
})
