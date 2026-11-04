import { defineConfig, devices } from '@playwright/test'

/**
 * DefectLens E2E 配置（T4.4b）。
 *
 * 前置条件（测试前手动准备，config 不自动起服务，见 e2e/README.md）：
 *   1) cd frontend && npm run build   # 生成 dist，由后端单进程托管
 *   2) 启动 uvicorn（http://localhost:3000）
 *
 * 单 worker：后端推理执行器并发上限为 2，串行执行避免互相抢占导致超时。
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 30_000 },
  retries: process.env.CI ? 1 : 0,
  reporter: [
    ['list'],
    ['html', { outputFolder: 'e2e/playwright-report', open: 'never' }],
    ['junit', { outputFile: 'e2e/test-results/junit.xml' }],
  ],
  outputDir: 'e2e/test-results',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'off',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
})
