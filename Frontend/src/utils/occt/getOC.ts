// src/utils/occt/getOC.ts
/* eslint-disable @typescript-eslint/no-explicit-any */

let ocPromise: Promise<any> | null = null

const nativeImport = new Function('u', 'return import(u)') as (u: string) => Promise<any>

export async function getOC(): Promise<any> {
  if (ocPromise) return ocPromise

  ocPromise = (async () => {
    const base = (import.meta.env.BASE_URL || '/').replace(/\/?$/, '/')

    await nativeImport(`${base}occt/occt-loader.mjs`)

    const factory = (globalThis as any).OpenCascade
    if (typeof factory !== 'function') {
      throw new Error('globalThis.OpenCascade is not a function (occt-loader did not set it)')
    }

    const oc = await factory({
      locateFile: (path: string) => `${base}occt/${path}`, 
    })

    return oc
  })()

  return ocPromise
}
