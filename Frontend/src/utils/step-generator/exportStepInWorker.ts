interface StepExportResponse {
  buffer?: ArrayBuffer
  error?: string
}

export function exportStepInWorker(
  chip: unknown,
  internalOnly: boolean = false,
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(
      new URL('./stepExport.worker.ts', import.meta.url),
      { type: 'module' },
    )

    worker.onmessage = (event: MessageEvent<StepExportResponse>) => {
      worker.terminate()

      if (event.data.error) {
        reject(new Error(event.data.error))
        return
      }
      if (!event.data.buffer) {
        reject(new Error('STEP worker returned no file data'))
        return
      }

      resolve(new Blob([event.data.buffer], { type: 'application/step' }))
    }

    worker.onerror = (event) => {
      worker.terminate()
      reject(new Error(event.message || 'STEP worker failed'))
    }

    worker.postMessage({ chip, internalOnly })
  })
}
