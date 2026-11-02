import { describe, expect, it } from 'vitest'

import { checkUploadFile, isAcceptedImage, MAX_UPLOAD_BYTES } from './uploadCheck'

const jpegFile = { name: 'steel.jpg', size: 1000, type: 'image/jpeg' }
const pngFile = { name: 'steel.png', size: 1000, type: 'image/png' }

describe('isAcceptedImage', () => {
  it('accepts JPEG and PNG MIME types', () => {
    expect(isAcceptedImage(jpegFile)).toBe(true)
    expect(isAcceptedImage(pngFile)).toBe(true)
  })

  it('accepts image extensions when the MIME type is empty', () => {
    expect(isAcceptedImage({ name: 'photo.JPG', size: 1000, type: '' })).toBe(true)
    expect(isAcceptedImage({ name: 'photo.jpeg', size: 1000, type: '' })).toBe(true)
    expect(isAcceptedImage({ name: 'photo.png', size: 1000, type: '' })).toBe(true)
  })

  it('rejects non-image MIME types even when the name looks like an image', () => {
    expect(isAcceptedImage({ name: 'notes.txt', size: 1000, type: 'text/plain' })).toBe(false)
    expect(isAcceptedImage({ name: 'fake.jpg', size: 1000, type: 'application/x-msdownload' })).toBe(false)
  })
})

describe('checkUploadFile', () => {
  it('accepts valid JPEG and PNG files', () => {
    expect(checkUploadFile(jpegFile)).toEqual({ ok: true })
    expect(checkUploadFile(pngFile)).toEqual({ ok: true })
  })

  it('rejects empty files with a friendly hint', () => {
    expect(checkUploadFile({ name: 'x.jpg', size: 0, type: 'image/jpeg' })).toEqual({
      ok: false,
      reason: 'empty',
      message: '文件内容为空，无法上传',
    })
  })

  it('rejects files above the size cap (backend answers 413 anyway)', () => {
    expect(
      checkUploadFile({ name: 'big.png', size: MAX_UPLOAD_BYTES + 1, type: 'image/png' }),
    ).toEqual({
      ok: false,
      reason: 'too-large',
      message: '图片超过 5.0 MB 大小上限，请压缩后再试',
    })
  })

  it('accepts a file exactly at the cap (boundary)', () => {
    expect(checkUploadFile({ name: 'max.png', size: MAX_UPLOAD_BYTES, type: 'image/png' })).toEqual({
      ok: true,
    })
  })

  it('rejects unsupported types with a friendly hint', () => {
    expect(checkUploadFile({ name: 'notes.txt', size: 100, type: 'text/plain' })).toEqual({
      ok: false,
      reason: 'unsupported-type',
      message: '仅支持 JPEG/PNG 格式的图片',
    })
  })

  it('rejects files with neither a name nor a MIME type', () => {
    expect(checkUploadFile({ name: '', size: 100, type: '' })).toEqual({
      ok: false,
      reason: 'unsupported-type',
      message: '仅支持 JPEG/PNG 格式的图片',
    })
  })

  it('checks the size cap before the type (mirrors backend 413-before-415 order)', () => {
    const result = checkUploadFile({
      name: 'huge.exe',
      size: MAX_UPLOAD_BYTES + 1,
      type: 'application/x-msdownload',
    })
    expect(result).toMatchObject({ ok: false, reason: 'too-large' })
  })
})
