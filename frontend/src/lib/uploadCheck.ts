/**
 * Client-side pre-flight for image uploads.
 *
 * Mirrors the backend's validation order (size cap first, then magic bytes)
 * so local rejections surface before the network round-trip. The backend
 * remains the source of truth — this only produces friendlier, faster errors.
 */

import { formatBytes } from './format'

export const MAX_UPLOAD_BYTES = 5 * 1024 * 1024

const ACCEPTED_IMAGE_TYPES: readonly string[] = ['image/jpeg', 'image/png']

const IMAGE_EXTENSION = /\.(jpe?g|png)$/i

export interface UploadFileLike {
  name: string
  size: number
  type: string
}

export type UploadRejectionReason = 'empty' | 'unsupported-type' | 'too-large'

export type UploadCheckResult =
  | { ok: true }
  | { ok: false; reason: UploadRejectionReason; message: string }

export function isAcceptedImage(file: UploadFileLike): boolean {
  if (file.type) return ACCEPTED_IMAGE_TYPES.includes(file.type)
  return IMAGE_EXTENSION.test(file.name)
}

export function checkUploadFile(file: UploadFileLike): UploadCheckResult {
  if (file.size <= 0) {
    return { ok: false, reason: 'empty', message: '文件内容为空，无法上传' }
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      ok: false,
      reason: 'too-large',
      message: `图片超过 ${formatBytes(MAX_UPLOAD_BYTES)} 大小上限，请压缩后再试`,
    }
  }
  if (!isAcceptedImage(file)) {
    return { ok: false, reason: 'unsupported-type', message: '仅支持 JPEG/PNG 格式的图片' }
  }
  return { ok: true }
}
