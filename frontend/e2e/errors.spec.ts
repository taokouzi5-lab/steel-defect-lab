import { expect, test } from '@playwright/test'

import { VALID_JPEG, makeTempFile, resultPanel } from './helpers'

const FAKE_JPEG_CONTENT = 'this is not a jpeg image, just text content with a .jpg suffix'

/**
 * 错误上传流程：文本内容改名 .jpg 的伪造图片能通过前端扩展名预检，
 * 由服务端魔术字节校验拒绝（415），前端归一化为友好中文「格式」提示，
 * 页面不崩溃且可恢复重试。
 */
test.describe('错误上传', () => {
  test('伪造 jpg 触发 415，展示友好中文格式错误且页面可恢复', async ({ page }) => {
    const fakeJpeg = makeTempFile('fake_steel.jpg', FAKE_JPEG_CONTENT)

    await page.goto('/')
    await page.locator('.upload-input').setInputFiles(fakeJpeg)
    await expect(page.locator('.upload-preview img')).toBeVisible()

    // 断言错误确实来自服务端 415（而非仅前端预检拦截）
    const [response] = await Promise.all([
      page.waitForResponse(
        (res) => res.url().includes('/api/v1/predict') && res.request().method() === 'POST',
      ),
      page.locator('.predict-button').click(),
    ])
    expect(response.status()).toBe(415)

    // 友好中文错误提示（前端 415 → unsupported_media_type 归一化消息）
    const alert = page.locator('.predict-error[role="alert"]')
    await expect(alert).toBeVisible()
    await expect(alert).toContainText('格式')

    // 页面不崩溃：主流程控件仍可用
    await expect(page.locator('h1')).toBeVisible()
    await expect(page.locator('.predict-button')).toBeEnabled()
    await expect(page.locator('.upload-preview img')).toBeVisible()

    // 错误后重试合法图片：状态恢复，结果面板出现，错误提示消失
    await page.locator('.upload-input').setInputFiles(VALID_JPEG)
    await page.locator('.predict-button').click()
    await expect(resultPanel(page)).toBeVisible()
    await expect(alert).toBeHidden()
  })
})
