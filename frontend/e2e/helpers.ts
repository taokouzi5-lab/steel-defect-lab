import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { expect } from '@playwright/test'
import type { Locator, Page } from '@playwright/test'

const E2E_DIR = fileURLToPath(new URL('.', import.meta.url))

/** 仓库根目录（frontend/e2e 上两级）。 */
export const REPO_ROOT = path.resolve(E2E_DIR, '..', '..')

/** 已入库的合法钢材图片（backend 测试资产）。 */
export const VALID_JPEG = path.join(REPO_ROOT, 'backend', 'tests', 'fixtures', 'valid_jpeg.jpg')

/** 视觉回归基线截图目录（入库，见 e2e/README.md 更新说明）。 */
export const SCREENSHOTS_DIR = path.join(E2E_DIR, 'screenshots')

/** 结果面板定位器（排除空态 is-empty 与推理中 is-pending）。 */
export const resultPanel = (scope: Page | Locator): Locator =>
  scope.locator('.result-panel:not(.is-empty):not(.is-pending)')

/** 在系统临时目录生成测试文件（伪装 jpg / 非法扩展名），由操作系统回收。 */
export function makeTempFile(name: string, content: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'defectlens-e2e-'))
  const filePath = path.join(dir, name)
  fs.writeFileSync(filePath, content)
  return filePath
}

/** 等待两帧，确保视口 resize 后布局稳定（不使用固定 sleep）。 */
export async function waitForStableLayout(page: Page): Promise<void> {
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
      }),
  )
}

/** 上传合法图片 → 点「开始推理」→ 等待结果面板出现并返回其定位器。 */
export async function runPredictWithValidImage(page: Page): Promise<Locator> {
  await page.locator('.upload-input').setInputFiles(VALID_JPEG)
  await expect(page.locator('.upload-preview img')).toBeVisible()
  const predictButton = page.locator('.predict-button')
  await expect(predictButton).toBeEnabled()
  await predictButton.click()
  const panel = resultPanel(page)
  await expect(panel).toBeVisible()
  return panel
}

/** 从「12.3%」这类文本解析 0-100 数值；解析失败返回 NaN。 */
export function parsePercentText(text: string | null): number {
  if (!text) return Number.NaN
  const match = /(-?\d+(?:\.\d+)?)\s*%/.exec(text)
  return match ? Number(match[1]) : Number.NaN
}

/** 从「推理 123.4 ms」或「推理 1.23 s」解析出毫秒数；解析失败返回 null。 */
export function parseLatencyMs(text: string | null): number | null {
  if (!text) return null
  const msMatch = /(\d+(?:\.\d+)?)\s*ms/.exec(text)
  if (msMatch) return Number(msMatch[1])
  const sMatch = /(\d+(?:\.\d+)?)\s*s\b/.exec(text)
  if (sMatch) return Number(sMatch[1]) * 1000
  return null
}
