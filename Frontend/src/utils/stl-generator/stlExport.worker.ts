/// <reference lib="webworker" />

import { exportStlFromChipJson } from '@/utils/stl-generator/exportStlFromChipJson'

interface StlExportRequest {
  chip: unknown
  internalOnly?: boolean
}

self.onmessage = async (event: MessageEvent<StlExportRequest>) => {
  try {
    const blob = await exportStlFromChipJson(event.data.chip, event.data.internalOnly)
    const buffer = await blob.arrayBuffer()
    self.postMessage({ buffer }, { transfer: [buffer] })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    self.postMessage({ error: message })
  }
}

export {}
