/**
 * Pure display helpers for DefectLens.
 *
 * Every function is defensive: non-finite numbers render as an em dash so a
 * malformed payload can never crash a page. Scales mirror the backend's
 * mace-f conventions (µg for per-inference carbon, phone-charge equivalence
 * at 0.1 gCO2eq/s).
 */

export interface CarbonFormat {
  text: string
  unit: 'g' | 'mg' | 'µg' | ''
}

const PHONE_CHARGE_GCO2EQ_PER_SECOND = 0.1

const PARAM_UNITS = [
  { threshold: 1e9, unit: 'G' },
  { threshold: 1e6, unit: 'M' },
  { threshold: 1e3, unit: 'K' },
] as const

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB'] as const

const CLASS_ZH: Record<string, string> = {
  crazing: '裂纹',
  inclusion: '夹杂',
  patches: '斑块',
  pitted_surface: '麻点',
  'rolled-in_scale': '氧化铁皮压入',
  scratches: '划痕',
}

export function clamp01(value: number): number {
  if (Number.isNaN(value)) return 0
  if (!Number.isFinite(value)) return value > 0 ? 1 : 0
  return Math.min(1, Math.max(0, value))
}

export function formatPercent(value: number, digits = 1): string {
  if (Number.isNaN(value)) return '—'
  return `${(clamp01(value) * 100).toFixed(digits)}%`
}

export function formatCarbon(value: number): CarbonFormat {
  if (!Number.isFinite(value)) return { text: '—', unit: '' }
  const magnitude = Math.abs(value)
  if (magnitude === 0) return { text: '0', unit: 'g' }
  if (magnitude < 1e-6) return { text: value.toExponential(2), unit: 'g' }
  if (magnitude < 1e-3) return { text: (magnitude * 1e6).toPrecision(3), unit: 'µg' }
  if (magnitude < 1) return { text: (magnitude * 1e3).toPrecision(3), unit: 'mg' }
  return { text: value.toPrecision(4), unit: 'g' }
}

export function formatMs(value: number): string {
  if (!Number.isFinite(value)) return '—'
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`
  return `${value.toFixed(1)} ms`
}

export function formatParams(value: number): string {
  if (!Number.isFinite(value)) return '—'
  for (const { threshold, unit } of PARAM_UNITS) {
    if (value >= threshold) return `${(value / threshold).toFixed(2)}${unit}`
  }
  return String(value)
}

function trimTrailingZeros(text: string): string {
  if (text.includes('e')) return text
  if (!text.includes('.')) return text
  return text.replace(/\.?0+$/, '')
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return '—'
  if (value === 0) return '0'
  const magnitude = Math.abs(value)
  if (magnitude >= 1e6 || magnitude < 1e-4) return value.toExponential(3)
  return trimTrailingZeros(value.toPrecision(4))
}

export function formatBytes(value: number): string {
  if (!Number.isFinite(value)) return '—'
  if (value === 0) return '0 B'
  let scaled = value
  let unitIndex = 0
  while (scaled >= 1024 && unitIndex < BYTE_UNITS.length - 1) {
    scaled /= 1024
    unitIndex += 1
  }
  const digits = unitIndex === 0 ? 0 : 1
  return `${scaled.toFixed(digits)} ${BYTE_UNITS[unitIndex]}`
}

export function carbonEquivalentText(co2eqG: number, backendEquivalent = ''): string {
  if (backendEquivalent.trim()) return backendEquivalent
  if (!Number.isFinite(co2eqG)) return '—'
  const seconds = co2eqG / PHONE_CHARGE_GCO2EQ_PER_SECOND
  let text: string
  if (seconds >= 100) text = seconds.toFixed(0)
  else if (seconds >= 1) text = seconds.toFixed(1)
  else if (seconds >= 0.01) text = seconds.toFixed(2)
  else text = trimTrailingZeros(seconds.toPrecision(3))
  return `≈ 手机充电 ${text} 秒`
}

function prettifyName(name: string): string {
  return name
    .split('_')
    .map((part) =>
      part
        .split('-')
        .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : word))
        .join('-'),
    )
    .join(' ')
}

export function classLabel(name: string): string {
  if (!name) return ''
  const zh = CLASS_ZH[name]
  return zh ? `${zh} ${prettifyName(name)}` : name
}
