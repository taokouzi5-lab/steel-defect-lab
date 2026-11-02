import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// RTL cannot register its auto-cleanup without vitest globals enabled.
afterEach(() => {
  cleanup()
})
