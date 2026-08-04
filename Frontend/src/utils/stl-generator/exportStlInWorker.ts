interface StlExportResponse {
  buffer?: ArrayBuffer
  error?: string
}

export function exportStlInWorker(
  chip: unknown,
  internalOnly: boolean = false,
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(
      new URL('./stlExport.worker.ts', import.meta.url),
      { type: 'module' },
    )

    worker.onmessage = (event: MessageEvent<StlExportResponse>) => {
      worker.terminate()

      if (event.data.error) {
        reject(new Error(event.data.error))
        return
      }
      if (!event.data.buffer) {
        reject(new Error('STL worker returned no file data'))
        return
      }

      resolve(new Blob([event.data.buffer], { type: 'model/stl' }))
    }

    worker.onerror = (event) => {
      worker.terminate()
      reject(new Error(event.message || 'STL worker failed'))
    }

    worker.postMessage({ chip, internalOnly })
  })
}
