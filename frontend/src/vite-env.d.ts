/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute backend origin; when unset the client uses same-origin relative paths (vite proxy). */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
