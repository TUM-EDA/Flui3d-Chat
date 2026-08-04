/// <reference lib="webworker" />

import { exportStepFromChipJson } from '@/utils/step-generator/exportStepFromChipJson'

interface StepExportRequest {
  chip: unknown
  internalOnly?: boolean
}

self.onmessage = async (event: MessageEvent<StepExportRequest>) => {
  try {
    const blob = await exportStepFromChipJson(event.data.chip, event.data.internalOnly)
    const buffer = await blob.arrayBuffer()
    self.postMessage({ buffer }, { transfer: [buffer] })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    self.postMessage({ error: message })
  }
}

export {}
