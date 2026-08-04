/* eslint-disable @typescript-eslint/no-explicit-any */

import type {
  Shape,
  CircleShape,
  LineShape,
  PolygonShape,
  CurveShape,
  ChamferShape,
  Point3,
} from '@/utils/step-generator/model/types'

import * as CAD from '@/utils/occt/cad'

function scaleP(p: Point3, zOffset: number, s: number): Point3 {
  return {
    x: p.x * s,
    y: p.y * s,
    z: (p.z * s) + zOffset,
  }
}

function buildCircleShape(oc: any, shape: CircleShape, zOffset: number, s: number) {
  let solid = CAD.makeCylinderZ(oc, shape.radius * s, shape.height * s)
  const base = scaleP(shape.center, zOffset, s)
  solid = CAD.translate(oc, solid, base.x, base.y, base.z)
  return solid
}

function buildLineShape(oc: any, shape: LineShape, zOffset: number, s: number) {
  const start = scaleP(shape.start, zOffset, s)
  const end = scaleP(shape.end, zOffset, s)
  const w = shape.width * s
  const h = shape.height * s

  return CAD.makeLineBox(
    oc,
    { x: start.x, y: start.y, z: start.z - h / 2 },
    { x: end.x, y: end.y, z: end.z - h / 2 },
    w,
    h
  )
}

function buildPolygonShape(oc: any, shape: PolygonShape, zOffset: number, s: number) {
  const pts = shape.points.map((p: Point3) => scaleP(p, zOffset, s))
  const scaledDir = { x: shape.direction.x * s, y: shape.direction.y * s, z: shape.direction.z * s }
  return CAD.makePolygonPrism(oc, pts, scaledDir)
}

function buildChamferShape(oc: any, shape: ChamferShape, zOffset: number, s: number) {
  let solid = CAD.makeConeZ(oc, shape.radius * s, shape.radius_top * s, shape.height * s)
  const base = scaleP(shape.center, zOffset, s)
  solid = CAD.translate(oc, solid, base.x, base.y, base.z)
  return solid
}

function isBridgeCurve(shape: CurveShape): boolean {
  const { start, end, center } = shape
  const zDiffStartEnd = Math.abs(start.z - end.z)
  const zDiffStartCenter = Math.abs(start.z - center.z)
  const zDiffEndCenter = Math.abs(end.z - center.z)

  const maxZDiff = Math.max(zDiffStartEnd, zDiffStartCenter, zDiffEndCenter)
  return maxZDiff > 1e-6
}

function buildCurveShape(oc: any, shape: CurveShape, zOffset: number, s: number) {
  const start = scaleP(shape.start, zOffset, s)
  const end = scaleP(shape.end, zOffset, s)
  const center = scaleP(shape.center, zOffset, s)

  let w = shape.width * s
  let h = shape.height * s

  if (isBridgeCurve(shape)) {
    w = shape.height * s
    h = shape.width * s
  }

  let solid = CAD.makeArcPipeRect(
    oc,
    start,
    end,
    center,
    w,
    h,
    shape.tangent
  )

  solid = CAD.translate(oc, solid, 0, 0, -h / 2)

  return solid
}

export function buildShapeToOcct(oc: any, shape: Shape, zOffset = 0, scale = 1) {
  switch (shape.type) {
    case 'Circle':
      return buildCircleShape(oc, shape, zOffset, scale)

    case 'Line':
      return buildLineShape(oc, shape, zOffset, scale)

    case 'Polygon':
      return buildPolygonShape(oc, shape, zOffset, scale)

    case 'Curve':
      return buildCurveShape(oc, shape, zOffset, scale)

    case 'Chamfer':
      return buildChamferShape(oc, shape, zOffset, scale)

    default:
      throw new Error(`Unsupported shape type: ${(shape as any)?.type}`)
  }
}
