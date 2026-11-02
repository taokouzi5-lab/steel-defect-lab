import { describe, expect, it } from 'vitest'

import {
  carbonEquivalentText,
  clamp01,
  classLabel,
  formatBytes,
  formatCarbon,
  formatMs,
  formatNumber,
  formatParams,
  formatPercent,
} from './format'

describe('clamp01', () => {
  it('passes values inside [0, 1] through unchanged', () => {
    expect(clamp01(0.5)).toBe(0.5)
    expect(clamp01(0)).toBe(0)
    expect(clamp01(1)).toBe(1)
  })

  it('clamps out-of-range values to the boundary', () => {
    expect(clamp01(1.5)).toBe(1)
    expect(clamp01(-0.2)).toBe(0)
  })

  it('maps NaN and Infinity to safe defaults', () => {
    expect(clamp01(Number.NaN)).toBe(0)
    expect(clamp01(Number.POSITIVE_INFINITY)).toBe(1)
  })
})

describe('formatPercent', () => {
  it('formats a confidence fraction as a percentage with one decimal', () => {
    expect(formatPercent(0.9944)).toBe('99.4%')
    expect(formatPercent(0.9831)).toBe('98.3%')
  })

  it('handles the 0 and 1 boundaries', () => {
    expect(formatPercent(0)).toBe('0.0%')
    expect(formatPercent(1)).toBe('100.0%')
  })

  it('defends against NaN with an em dash', () => {
    expect(formatPercent(Number.NaN)).toBe('—')
  })

  it('clamps values beyond 100% instead of overflowing bars', () => {
    expect(formatPercent(1.05)).toBe('100.0%')
    expect(formatPercent(-0.1)).toBe('0.0%')
  })
})

describe('formatCarbon', () => {
  it('shows the microgram scale for typical per-inference values', () => {
    expect(formatCarbon(7.399e-5)).toEqual({ text: '74.0', unit: 'µg' })
  })

  it('formats zero as plain grams', () => {
    expect(formatCarbon(0)).toEqual({ text: '0', unit: 'g' })
  })

  it('defends against NaN and Infinity', () => {
    expect(formatCarbon(Number.NaN)).toEqual({ text: '—', unit: '' })
    expect(formatCarbon(Number.POSITIVE_INFINITY)).toEqual({ text: '—', unit: '' })
  })

  it('switches to the milligram scale between 1e-3 and 1 g', () => {
    expect(formatCarbon(0.308)).toEqual({ text: '308', unit: 'mg' })
  })

  it('keeps the gram scale with four significant digits at or above 1 g', () => {
    expect(formatCarbon(2.5)).toEqual({ text: '2.500', unit: 'g' })
  })

  it('falls back to scientific notation below one microgram', () => {
    expect(formatCarbon(7.4e-7)).toEqual({ text: '7.40e-7', unit: 'g' })
  })

  it('formats negative values by magnitude (defensive)', () => {
    expect(formatCarbon(-0.0015)).toEqual({ text: '1.50', unit: 'mg' })
  })
})

describe('formatMs', () => {
  it('formats millisecond latencies with one decimal', () => {
    expect(formatMs(9.317)).toBe('9.3 ms')
    expect(formatMs(11.449)).toBe('11.4 ms')
  })

  it('switches to seconds at one second', () => {
    expect(formatMs(1250)).toBe('1.25 s')
  })

  it('defends against NaN', () => {
    expect(formatMs(Number.NaN)).toBe('—')
  })

  it('formats zero', () => {
    expect(formatMs(0)).toBe('0.0 ms')
  })
})

describe('formatParams', () => {
  it('formats model parameter counts in millions', () => {
    expect(formatParams(11179590)).toBe('11.18M')
    expect(formatParams(23520326)).toBe('23.52M')
  })

  it('uses kilo and giga scales', () => {
    expect(formatParams(1500)).toBe('1.50K')
    expect(formatParams(2500000000)).toBe('2.50G')
  })

  it('keeps small counts plain', () => {
    expect(formatParams(950)).toBe('950')
  })

  it('defends against NaN', () => {
    expect(formatParams(Number.NaN)).toBe('—')
  })
})

describe('formatNumber', () => {
  it('keeps everyday numbers readable', () => {
    expect(formatNumber(231)).toBe('231')
    expect(formatNumber(0.3085)).toBe('0.3085')
    expect(formatNumber(1.39)).toBe('1.39')
  })

  it('uses scientific notation for tiny benchmark energies', () => {
    expect(formatNumber(4.5748e-5)).toBe('4.575e-5')
  })

  it('uses scientific notation for very large numbers', () => {
    expect(formatNumber(11179590)).toBe('1.118e+7')
  })

  it('handles zero and NaN', () => {
    expect(formatNumber(0)).toBe('0')
    expect(formatNumber(Number.NaN)).toBe('—')
  })
})

describe('formatBytes', () => {
  it('formats the upload cap in megabytes', () => {
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB')
  })

  it('handles kilobytes and bytes', () => {
    expect(formatBytes(1536)).toBe('1.5 KB')
    expect(formatBytes(512)).toBe('512 B')
  })

  it('formats zero', () => {
    expect(formatBytes(0)).toBe('0 B')
  })

  it('defends against NaN', () => {
    expect(formatBytes(Number.NaN)).toBe('—')
  })
})

describe('carbonEquivalentText', () => {
  it('prefers the backend-provided equivalent text', () => {
    expect(carbonEquivalentText(0.74, '≈ 手机充电 0.74 秒')).toBe('≈ 手机充电 0.74 秒')
  })

  it('computes the phone-charging seconds from co2eq when missing', () => {
    expect(carbonEquivalentText(0.74, '')).toBe('≈ 手机充电 7.4 秒')
    expect(carbonEquivalentText(7.5, '')).toBe('≈ 手机充电 75.0 秒')
  })

  it('rounds large second counts to integers like the backend', () => {
    expect(carbonEquivalentText(10, '')).toBe('≈ 手机充电 100 秒')
  })

  it('uses three significant digits below one hundredth of a second', () => {
    expect(carbonEquivalentText(7.4e-5, '')).toBe('≈ 手机充电 0.00074 秒')
  })

  it('defends against NaN', () => {
    expect(carbonEquivalentText(Number.NaN, '')).toBe('—')
  })
})

describe('classLabel', () => {
  it('maps the six NEU-DET classes to bilingual labels', () => {
    expect(classLabel('crazing')).toBe('裂纹 Crazing')
    expect(classLabel('inclusion')).toBe('夹杂 Inclusion')
    expect(classLabel('patches')).toBe('斑块 Patches')
    expect(classLabel('pitted_surface')).toBe('麻点 Pitted Surface')
    expect(classLabel('rolled-in_scale')).toBe('氧化铁皮压入 Rolled-In Scale')
    expect(classLabel('scratches')).toBe('划痕 Scratches')
  })

  it('falls back to the raw name for unknown classes', () => {
    expect(classLabel('mystery_class')).toBe('mystery_class')
  })

  it('handles empty input', () => {
    expect(classLabel('')).toBe('')
  })
})
