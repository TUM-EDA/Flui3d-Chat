/* eslint-disable @typescript-eslint/no-explicit-any */

import type { ChipJSON, Point3 } from '@/utils/step-generator/model/types'
import { getOC } from '@/utils/occt/getOC'
import * as CAD from '@/utils/occt/cad'
import { buildShapeToOcct } from '@/utils/step-generator/shapeToOcct'

export async function buildOcctShapeFromChipJson(
  chip: any,
  internalOnly: boolean = false,
): Promise<{ oc: any; shape: any }> {
  const oc = await getOC()

  // Support both general nested format (original ChipJSON) and flat format (serialized com.tdmp.Chip)
  const length = chip.general ? chip.general.length : chip.length
  const width = chip.general ? chip.general.width : chip.width
  const thickness = chip.general ? chip.general.thickness : chip.thickness

  // Support both crosslayerConnections and crossLayerConnections
  const crosslayerConnections = chip.crosslayerConnections || chip.crossLayerConnections || []

  // Fluid3D 内部用 µm，STEP 标准用 mm，缩放 1/1000
  const S = 1 / 1000

  // chip base
  let result: any = null;
  if (!internalOnly) {
    result = CAD.makeBox(
      oc,
      length * S,
      width * S,
      thickness * S
    )
    console.log('[STEP] base box:', length * S, '×', width * S, '×', thickness * S)
  }

  let shapeIdx = 0
  const negativeShapes: any[] = []
  const positiveShapes: any[] = []

  function collectShape(label: string, shape: any) {
    try {
      const solid = buildShapeToOcct(oc, shape, 0, S)
      
      const fillValue = shape.fill
      const shouldSubtract = fillValue === undefined ? true : !fillValue
      
      if (shouldSubtract) {
        negativeShapes.push(solid)
      } else {
        positiveShapes.push(solid)
      }
      
      console.log(`[STEP] #${shapeIdx} ${label} type=${shape.type} subtract=${shouldSubtract} -> solid OK`)
    } catch (e) {
      console.error(`[STEP] #${shapeIdx} ${label} type=${shape.type} BUILD FAILED:`, e)
    }
    shapeIdx++
  }

  for (const [li, layer] of chip.layers.entries()) {
    for (const [ci, channel] of layer.channels.entries()) {
      for (const shape of channel.shapes) {
        collectShape(`layer[${li}].channel[${ci}]`, shape)
      }
    }

    for (const [ci, comp] of layer.components.entries()) {
      for (const shape of comp.shapes) {
        collectShape(`layer[${li}].comp[${ci}].shape`, shape)
      }

      for (const [chi, ch] of comp.channels.entries()) {
        for (const shape of ch.shapes) {
          collectShape(`layer[${li}].comp[${ci}].channel[${chi}]`, shape)
        }
      }
    }

    if (layer.compensation) {
      try {
        const compSolid = CAD.makePolygonPrism(
          oc,
          layer.compensation.points.map((p: Point3) => ({
            x: p.x * S,
            y: p.y * S,
            z: p.z * S,
          })),
          { x: layer.compensation.direction.x * S, y: layer.compensation.direction.y * S, z: layer.compensation.direction.z * S }
        )
        const fillValue = layer.compensation.fill
        const shouldSubtract = fillValue === undefined ? true : !fillValue
        if (shouldSubtract) {
          negativeShapes.push(compSolid)
        } else {
          positiveShapes.push(compSolid)
        }
        console.log(`[STEP] layer[${li}] compensation collected OK (subtract=${shouldSubtract})`)
      } catch (e) {
        console.error(`[STEP] layer[${li}] compensation FAILED:`, e)
      }
    }
  }

  for (const [ci, conn] of crosslayerConnections.entries()) {
    for (const shape of conn.shapes) {
      collectShape(`crosslayer[${ci}]`, shape)
    }
  }

  console.log(`[STEP] Shapes collected: ${negativeShapes.length} negative, ${positiveShapes.length} positive`)

  // 1. Subtract the negative shapes (holes/channels)
  if (negativeShapes.length > 0) {
    try {
      console.log('[STEP] Fusing all negative tool shapes into a single valid solid...')
      let masterTool = CAD.fuseTree(oc, negativeShapes)

      if (internalOnly) {
        result = masterTool
      } else {
        console.log('[STEP] Executing monolithic Cut with the fused tool...')
        result = CAD.cut(oc, result, masterTool)
      }
    } catch (e) {
      console.error('[STEP] Fusing or Cut of negative shapes FAILED:', e)
    }
  }

  // 2. Add back the positive shapes (pillars)
  if (positiveShapes.length > 0) {
    try {
      console.log('[STEP] Fusing all positive tool shapes into a single valid solid...')
      let masterAdd = CAD.fuseTree(oc, positiveShapes)

      if (internalOnly) {
        if (result) {
          console.log('[STEP] Subtracting positive features (pillars) from the internal volume...')
          result = CAD.cut(oc, result, masterAdd)
        } else {
          result = masterAdd
        }
      } else {
        console.log('[STEP] Fusing positive features back to the main block...')
        result = CAD.fuse(oc, result, masterAdd)
      }
    } catch (e) {
      console.error('[STEP] Fusing or Add of positive shapes FAILED:', e)
    }
  }

  if (!result) {
    throw new Error('No shapes to export')
  }

  try {
    console.log('[STEP] Executing topological cleanup (UnifySameDomain)...')
    result = CAD.unifyShape(oc, result)
    console.log('[STEP] Shape processing complete.')
  } catch (e) {
    console.error('[STEP] Topological cleanup FAILED:', e)
  }

  return { oc, shape: result }
}

export async function exportStepFromChipJson(chip: any, internalOnly: boolean = false): Promise<Blob> {
  const { oc, shape } = await buildOcctShapeFromChipJson(chip, internalOnly)
  return CAD.exportSTEP(oc, shape, 'fluid3d.step')
}
