/* eslint-disable @typescript-eslint/no-explicit-any */

import * as CAD from '@/utils/occt/cad'
import { buildOcctShapeFromChipJson } from '@/utils/step-generator/exportStepFromChipJson'

export async function exportStlFromChipJson(
  chip: any,
  internalOnly: boolean = false,
): Promise<Blob> {
  const { oc, shape } = await buildOcctShapeFromChipJson(chip, internalOnly)
  return CAD.exportSTL(oc, shape, 'fluid3d.stl')
}
