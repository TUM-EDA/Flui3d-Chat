/* eslint-disable @typescript-eslint/no-explicit-any */

function ensureDir(oc: any, dir: string) {
  try {
    oc.FS.stat(dir)
  } catch {
    oc.FS.mkdir(dir)
  }
}

function toValue(x: any) {
  return x && typeof x === 'object' && 'value' in x ? (x as any).value : x
}

function pickStepModelType(oc: any) {
  const t = oc.STEPControl_StepModelType
  return t?.STEPControl_AsIs ?? t?.STEPControl_ManifoldSolidBrep ?? 0
}

function makeProgress(oc: any) {
  try {
    if (oc.Message_ProgressRange_1) return new oc.Message_ProgressRange_1()
    if (oc.Message_ProgressRange) return new oc.Message_ProgressRange()
  } catch {
    // ignore
  }
  return undefined
}

function stepTransfer(writer: any, shape: any, mode: any, progress: any) {
  try {
    return writer.Transfer(shape, mode, true, progress)
  } catch {
    try {
      return writer.Transfer(shape, mode, false, progress)
    } catch {
      return writer.Transfer(shape, mode, true)
    }
  }
}

function matchExact(key: string, prefix: string): boolean {
  if (key === prefix) return true
  if (key.startsWith(prefix + '_')) return true
  return false
}

function norm3(x: number, y: number, z: number) {
  return Math.sqrt(x * x + y * y + z * z)
}

function cross3(
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number
) {
  return {
    x: ay * bz - az * by,
    y: az * bx - ax * bz,
    z: ax * by - ay * bx,
  }
}

function probeNoArg(oc: any, prefix: string) {
  const names = Object.keys(oc).filter(k => matchExact(k, prefix)).sort()
  for (const n of names) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    try {
      const obj = new Ctor()
      if (obj) return obj
    } catch {
      // wrong overload, try next
    }
  }
  throw new Error(`No ${prefix} no-arg constructor found`)
}

function probeXYZ(oc: any, prefix: string, x: number, y: number, z: number) {
  const names = Object.keys(oc).filter(k => matchExact(k, prefix)).sort()
  for (const n of names) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    try {
      const obj = new Ctor(x, y, z)
      if (obj) return obj
    } catch {
      // wrong overload, try next
    }
  }
  if (typeof oc[prefix] === 'function') {
    return new oc[prefix](x, y, z)
  }
  throw new Error(`No ${prefix} overload accepts (x, y, z)`)
}

export function pnt(oc: any, x: number, y: number, z: number) {
  return probeXYZ(oc, 'gp_Pnt', x, y, z)
}

export function vec(oc: any, x: number, y: number, z: number) {
  return probeXYZ(oc, 'gp_Vec', x, y, z)
}

export function dir(oc: any, x: number, y: number, z: number) {
  return probeXYZ(oc, 'gp_Dir', x, y, z)
}

export function makeBox(oc: any, dx: number, dy: number, dz: number) {
  const names = Object.keys(oc).filter(k => k.startsWith('BRepPrimAPI_MakeBox_')).sort()
  for (const n of names) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    try {
      const inst = new Ctor(dx, dy, dz)
      const s = inst?.Shape?.()
      if (s) return s
    } catch {
      // ignore
    }
  }
  throw new Error('MakeBox failed')
}

export function makeCylinderZ(oc: any, radius: number, height: number) {
  const names = Object.keys(oc).filter(k => k.startsWith('BRepPrimAPI_MakeCylinder_')).sort()
  for (const n of names) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    try {
      const inst = new Ctor(radius, height)
      const s = inst?.Shape?.()
      if (s) return s
    } catch {
      // ignore
    }
  }
  throw new Error('MakeCylinder failed')
}

export function makeConeZ(
  oc: any,
  radiusBottom: number,
  radiusTop: number,
  height: number
) {
  const names = Object.keys(oc).filter(k => k.startsWith('BRepPrimAPI_MakeCone_')).sort()
  for (const n of names) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    try {
      const inst = new Ctor(radiusBottom, radiusTop, height)
      const s = inst?.Shape?.()
      if (s) return s
    } catch {
      // ignore
    }
  }
  throw new Error('MakeCone failed')
}

export function translate(oc: any, shape: any, dx: number, dy: number, dz: number) {
  const trsf = probeNoArg(oc, 'gp_Trsf')
  const v = vec(oc, dx, dy, dz)

  const candidates = [
    'SetTranslation_2', 'SetTranslation_1', 'SetTranslation_3',
    'SetTranslation_4', 'SetTranslation',
  ]
  let translated = false
  for (const m of candidates) {
    if (typeof trsf[m] !== 'function') continue
    try {
      trsf[m](v)
      translated = true
      break
    } catch {
      // wrong overload, try next
    }
  }
  if (!translated) {
    throw new Error('gp_Trsf: no SetTranslation overload accepted gp_Vec')
  }

  const names = Object.keys(oc).filter(k => k.startsWith('BRepBuilderAPI_Transform_')).sort()
  for (const n of names) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    try {
      const tx = new Ctor(shape, trsf, true)
      const s = tx?.Shape?.()
      if (s) return s
    } catch {
      // try next
    }
  }
  throw new Error('BRepBuilderAPI_Transform failed')
}

export function transform(oc: any, shape: any, trsf: any) {
  const names = Object.keys(oc).filter(k => matchExact(k, 'BRepBuilderAPI_Transform')).sort()
  for (const n of names) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    try {
      const tx = new Ctor(shape, trsf, true)
      const s = tx?.Shape?.()
      if (s) return s
    } catch {
      // try next
    }
  }
  throw new Error('BRepBuilderAPI_Transform failed')
}

function callBuild(op: any) {
  const candidates = ['Build_1', 'Build_2', 'Build']
  for (const m of candidates) {
    if (typeof op[m] === 'function') {
      try { op[m](); return } catch { /* next */ }
    }
  }
}

function callShape(op: any): any {
  const candidates = ['Shape', 'Shape_1', 'Shape_2']
  for (const m of candidates) {
    if (typeof op[m] === 'function') {
      try {
        const s = op[m]()
        if (s) return s
      } catch { /* next */ }
    }
  }
  return null
}

export function fuse(oc: any, a: any, b: any) {
  const names = Object.keys(oc).filter(k => matchExact(k, 'BRepAlgoAPI_Fuse')).sort()
  const progress = makeProgress(oc)
  for (const n of names) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    if (progress) {
      try {
        const op = new Ctor(a, b, progress)
        callBuild(op)
        const s = callShape(op)
        if (s) return s
      } catch { /* next */ }
    }
    try {
      const op = new Ctor(a, b)
      callBuild(op)
      const s = callShape(op)
      if (s) return s
    } catch { /* next */ }
  }
  throw new Error('BRepAlgoAPI_Fuse failed')
}

export function fuseTree(oc: any, shapes: any[]): any {
  if (!shapes || shapes.length === 0) return null
  if (shapes.length === 1) return shapes[0]

  let current = [...shapes]
  let iteration = 0
  
  while (current.length > 1) {
    const next = []
    const t0 = performance.now()
    
    for (let i = 0; i < current.length; i += 2) {
      if (i + 1 < current.length) {
        try {
          next.push(fuse(oc, current[i], current[i + 1]))
        } catch (e) {
          console.warn(`[STEP] fuseTree pair failed at iteration ${iteration}, fallback to keep both:`, e)
          next.push(current[i])
          next.push(current[i + 1])
        }
      } else {
        next.push(current[i])
      }
    }
    
    const t1 = performance.now()
    console.log(`[STEP] fuseTree iteration ${iteration}: reduced ${current.length} -> ${next.length} shapes in ${(t1 - t0).toFixed(1)}ms`)
    
    if (next.length === current.length) {
      console.error('[STEP] fuseTree made no progress! Fallback to linear fuse.')
      let master = current[0]
      for (let i = 1; i < current.length; i++) {
        try { master = fuse(oc, master, current[i]) } catch (e) { console.warn('[STEP] linear fuse fallback failed:', e) }
      }
      return master
    }
    current = next
    iteration++
  }
  return current[0]
}

export function makeCompound(oc: any, shapes: any[]): any {
  if (shapes.length === 0) throw new Error('makeCompound: empty shapes')
  if (shapes.length === 1) return shapes[0]

  const compoundNames = Object.keys(oc).filter(k => matchExact(k, 'TopoDS_Compound')).sort()
  let compound: any = null
  for (const n of compoundNames) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    try { compound = new Ctor(); if (compound) break } catch { /* next */ }
  }
  if (!compound) throw new Error('TopoDS_Compound construction failed')

  const builderNames = Object.keys(oc).filter(k => matchExact(k, 'BRep_Builder')).sort()
  let builder: any = null
  for (const n of builderNames) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    try { builder = new Ctor(); if (builder) break } catch { /* next */ }
  }
  if (!builder) throw new Error('BRep_Builder construction failed')

  const makeCandidates = ['MakeCompound', 'MakeCompound_1', 'MakeCompound_2']
  let made = false
  for (const m of makeCandidates) {
    if (typeof builder[m] !== 'function') continue
    try { builder[m](compound); made = true; break } catch { /* next */ }
  }
  if (!made) throw new Error('BRep_Builder.MakeCompound failed')

  const addCandidates = ['Add', 'Add_1', 'Add_2', 'Add_3']
  for (const shape of shapes) {
    let added = false
    for (const m of addCandidates) {
      if (typeof builder[m] !== 'function') continue
      try { builder[m](compound, shape); added = true; break } catch { /* next */ }
    }
    if (!added) throw new Error('BRep_Builder.Add failed')
  }

  return compound
}

export function cut(oc: any, a: any, b: any) {
  const names = Object.keys(oc).filter(k => matchExact(k, 'BRepAlgoAPI_Cut')).sort()
  const progress = makeProgress(oc)
  for (const n of names) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    if (progress) {
      try {
        const op = new Ctor(a, b, progress)
        callBuild(op)
        const s = callShape(op)
        if (s) return s
      } catch { /* next */ }
    }
    try {
      const op = new Ctor(a, b)
      callBuild(op)
      const s = callShape(op)
      if (s) return s
    } catch { /* next */ }
  }
  throw new Error('BRepAlgoAPI_Cut failed')
}

export function makeLineBox(
  oc: any,
  start: { x: number; y: number; z: number },
  end: { x: number; y: number; z: number },
  width: number,
  height: number
) {
  const dx = end.x - start.x
  const dy = end.y - start.y
  const dz = end.z - start.z
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz)

  if (len <= 1e-9) {
    throw new Error('LineShape has zero length')
  }

  const ex = { x: dx / len, y: dy / len, z: dz / len }

  let widthDir = cross3(0, 0, 1, ex.x, ex.y, ex.z)
  let nW = norm3(widthDir.x, widthDir.y, widthDir.z)
  if (nW < 1e-6) {
    widthDir = { x: 1, y: 0, z: 0 }
  } else {
    widthDir = { x: widthDir.x / nW, y: widthDir.y / nW, z: widthDir.z / nW }
  }

  let heightDir = cross3(ex.x, ex.y, ex.z, widthDir.x, widthDir.y, widthDir.z)
  let nH = norm3(heightDir.x, heightDir.y, heightDir.z)
  heightDir = { x: heightDir.x / nH, y: heightDir.y / nH, z: heightDir.z / nH }

  const hw = width / 2
  const hh = height

  const p1 = {
    x: start.x - widthDir.x * hw,
    y: start.y - widthDir.y * hw,
    z: start.z - widthDir.z * hw,
  }
  const p2 = {
    x: start.x + widthDir.x * hw,
    y: start.y + widthDir.y * hw,
    z: start.z + widthDir.z * hw,
  }
  const p3 = {
    x: p2.x + heightDir.x * hh,
    y: p2.y + heightDir.y * hh,
    z: p2.z + heightDir.z * hh,
  }
  const p4 = {
    x: p1.x + heightDir.x * hh,
    y: p1.y + heightDir.y * hh,
    z: p1.z + heightDir.z * hh,
  }

  return makePolygonPrism(oc, [p1, p2, p3, p4], { x: dx, y: dy, z: dz })
}

export function makePolygonPrism(
  oc: any,
  points: Array<{ x: number; y: number; z: number }>,
  directionVec: { x: number; y: number; z: number }
) {
  if (!points || points.length < 3) {
    throw new Error('PolygonShape needs at least 3 points')
  }

  const wireMaker = probeNoArg(oc, 'BRepBuilderAPI_MakePolygon')

  for (const pt of points) {
    const p = pnt(oc, pt.x, pt.y, pt.z)
    const addCandidates = ['Add_1', 'Add_2', 'Add_3', 'Add']
    let added = false
    for (const m of addCandidates) {
      if (typeof wireMaker[m] !== 'function') continue
      try { wireMaker[m](p); added = true; break } catch { /* next */ }
    }
    if (!added) throw new Error('BRepBuilderAPI_MakePolygon: no Add overload accepted gp_Pnt')
  }

  const closeCandidates = ['Close', 'Close_1', 'Close_2']
  for (const m of closeCandidates) {
    if (typeof wireMaker[m] === 'function') {
      try { wireMaker[m](); break } catch { /* next */ }
    }
  }

  const wireCandidates = ['Wire', 'Wire_1', 'Wire_2']
  let wire: any = null
  for (const m of wireCandidates) {
    if (typeof wireMaker[m] !== 'function') continue
    try { wire = wireMaker[m](); if (wire) break } catch { /* next */ }
  }
  if (!wire) throw new Error('BRepBuilderAPI_MakePolygon: Wire() failed')

  const faceNames = Object.keys(oc).filter(k => matchExact(k, 'BRepBuilderAPI_MakeFace')).sort()
  let face: any = null
  const faceTries = [
    (Ctor: any) => new Ctor(wire, true),
    (Ctor: any) => new Ctor(wire, false),
    (Ctor: any) => new Ctor(wire),
  ]
  outer: for (const n of faceNames) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    for (const tryFn of faceTries) {
      try {
        const fm = tryFn(Ctor)
        const f = fm?.Face?.() ?? fm?.Face_1?.() ?? fm?.Face_2?.()
        if (f) { face = f; break outer }
      } catch { /* next */ }
    }
  }
  if (!face) throw new Error('BRepBuilderAPI_MakeFace failed')

  const v = vec(oc, directionVec.x, directionVec.y, directionVec.z)
  const prismNames = Object.keys(oc).filter(k => matchExact(k, 'BRepPrimAPI_MakePrism')).sort()
  for (const n of prismNames) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    try {
      const prism = new Ctor(face, v, true, true)
      const s = prism?.Shape?.()
      if (s) return s
    } catch { /* next */ }
    try {
      const prism = new Ctor(face, v)
      const s = prism?.Shape?.()
      if (s) return s
    } catch { /* next */ }
  }
  throw new Error('BRepPrimAPI_MakePrism failed')
}

export function makePolygonFace(
  oc: any,
  points: Array<{ x: number; y: number; z: number }>
) {
  if (!points || points.length < 3) {
    throw new Error('makePolygonFace requires at least 3 points')
  }

  const wireMaker = probeNoArg(oc, 'BRepBuilderAPI_MakePolygon')

  for (const pt of points) {
    const p = pnt(oc, pt.x, pt.y, pt.z)
    const addCandidates = ['Add_1', 'Add_2', 'Add_3', 'Add']
    let added = false
    for (const m of addCandidates) {
      if (typeof wireMaker[m] !== 'function') continue
      try { wireMaker[m](p); added = true; break } catch { /* next */ }
    }
    if (!added) throw new Error('makePolygonFace: no Add overload accepted gp_Pnt')
  }

  const closeCandidates = ['Close', 'Close_1', 'Close_2']
  for (const m of closeCandidates) {
    if (typeof wireMaker[m] === 'function') {
      try { wireMaker[m](); break } catch { /* next */ }
    }
  }

  const wireCandidates = ['Wire', 'Wire_1', 'Wire_2']
  let wire: any = null
  for (const m of wireCandidates) {
    if (typeof wireMaker[m] !== 'function') continue
    try { wire = wireMaker[m](); if (wire) break } catch { /* next */ }
  }
  if (!wire) throw new Error('makePolygonFace: Wire() failed')

  const faceNames = Object.keys(oc).filter(k => matchExact(k, 'BRepBuilderAPI_MakeFace')).sort()
  let face: any = null
  const faceTries = [
    (Ctor: any) => new Ctor(wire, true),
    (Ctor: any) => new Ctor(wire, false),
    (Ctor: any) => new Ctor(wire),
  ]
  outer: for (const n of faceNames) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    for (const tryFn of faceTries) {
      try {
        const fm = tryFn(Ctor)
        const f = fm?.Face?.() ?? fm?.Face_1?.() ?? fm?.Face_2?.()
        if (f) { face = f; break outer }
      } catch { /* next */ }
    }
  }
  if (!face) throw new Error('makePolygonFace: MakeFace failed')

  return face
}

export function makeArcPipeRect(
  oc: any,
  start: { x: number; y: number; z: number },
  end: { x: number; y: number; z: number },
  center: { x: number; y: number; z: number },
  width: number,
  height: number,
  tangent?: { x: number; y: number; z: number }
) {
  function dot3(ax: number, ay: number, az: number, bx: number, by: number, bz: number) {
    return ax * bx + ay * by + az * bz;
  }

  const v1 = { x: start.x - center.x, y: start.y - center.y, z: start.z - center.z };
  const v2 = { x: end.x - center.x, y: end.y - center.y, z: end.z - center.z };
  
  let axDir = cross3(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
  let nAx = norm3(axDir.x, axDir.y, axDir.z);

  if (nAx < 1e-6 && tangent) {
    axDir = cross3(v1.x, v1.y, v1.z, tangent.x, tangent.y, tangent.z);
    nAx = norm3(axDir.x, axDir.y, axDir.z);
  }

  if (nAx < 1e-6) {
    axDir = { x: 0, y: 0, z: 1 };
    nAx = 1;
  }
  axDir = { x: axDir.x / nAx, y: axDir.y / nAx, z: axDir.z / nAx };

  let c12 = cross3(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
  const sinTheta = dot3(c12.x, c12.y, c12.z, axDir.x, axDir.y, axDir.z) / (norm3(v1.x, v1.y, v1.z) * norm3(v2.x, v2.y, v2.z));
  const cosTheta = dot3(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z) / (norm3(v1.x, v1.y, v1.z) * norm3(v2.x, v2.y, v2.z));
  
  let revolAngle = Math.atan2(Math.abs(sinTheta), cosTheta);
  if (revolAngle < 1e-6) revolAngle = 2 * Math.PI;

  let t_start = cross3(axDir.x, axDir.y, axDir.z, v1.x, v1.y, v1.z);
  let nT = norm3(t_start.x, t_start.y, t_start.z);
  t_start = { x: t_start.x / nT, y: t_start.y / nT, z: t_start.z / nT };

  let widthDir = cross3(0, 0, 1, t_start.x, t_start.y, t_start.z);
  let nW = norm3(widthDir.x, widthDir.y, widthDir.z);
  if (nW < 1e-6) {
    widthDir = { x: 1, y: 0, z: 0 };
  } else {
    widthDir = { x: widthDir.x / nW, y: widthDir.y / nW, z: widthDir.z / nW };
  }

  let heightDir = cross3(t_start.x, t_start.y, t_start.z, widthDir.x, widthDir.y, widthDir.z);
  let nH = norm3(heightDir.x, heightDir.y, heightDir.z);
  heightDir = { x: heightDir.x / nH, y: heightDir.y / nH, z: heightDir.z / nH };

  const centerPnt = pnt(oc, center.x, center.y, center.z);
  const ax1Dir = dir(oc, axDir.x, axDir.y, axDir.z);

  const ax1Names = Object.keys(oc).filter(k => matchExact(k, 'gp_Ax1')).sort();
  let ax1: any = null;
  for (const n of ax1Names) {
    const Ctor = oc[n];
    if (typeof Ctor !== 'function') continue;
    try {
      ax1 = new Ctor(centerPnt, ax1Dir);
      if (ax1) { break; }
    } catch { /* next */ }
  }
  if (!ax1) throw new Error('gp_Ax1 construction failed');

  const hw = width / 2;
  const hh = height;

  const p1 = {
    x: start.x - widthDir.x * hw,
    y: start.y - widthDir.y * hw,
    z: start.z - widthDir.z * hw,
  };
  const p2 = {
    x: start.x + widthDir.x * hw,
    y: start.y + widthDir.y * hw,
    z: start.z + widthDir.z * hw,
  };
  const p3 = {
    x: p2.x + heightDir.x * hh,
    y: p2.y + heightDir.y * hh,
    z: p2.z + heightDir.z * hh,
  };
  const p4 = {
    x: p1.x + heightDir.x * hh,
    y: p1.y + heightDir.y * hh,
    z: p1.z + heightDir.z * hh,
  };

  const profileFace = makePolygonFace(oc, [p1, p2, p3, p4]);

  const revolNames = Object.keys(oc)
    .filter(k => k.startsWith('BRepPrimAPI_MakeRevol_'))
    .sort();

  let result: any = null;

  for (const n of revolNames) {
    const Ctor = oc[n];
    if (typeof Ctor !== 'function') continue;
    try {
      const inst = new Ctor(profileFace, ax1, revolAngle, true);
      const s = inst?.Shape?.()
      if (s) { result = s; break; }
    } catch { /* next */ }
    try {
      const inst = new Ctor(profileFace, ax1, revolAngle);
      const s = inst?.Shape?.()
      if (s) { result = s; break; }
    } catch { /* next */ }
  }

  if (!result) {
    throw new Error('BRepPrimAPI_MakeRevol failed');
  }

  return result;
}

export function unifyShape(oc: any, shape: any): any {
  const unifyNames = Object.keys(oc).filter(k => matchExact(k, 'ShapeUpgrade_UnifySameDomain')).sort()
  for (const n of unifyNames) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    try {
      const unifier = new Ctor(shape, true, true, true)
      if (typeof unifier.Build === 'function') unifier.Build()
      else if (typeof unifier.Build_1 === 'function') unifier.Build_1()
      const result = unifier.Shape?.() ?? unifier.Shape_1?.()
      if (result) return result
    } catch { /* next */ }
    try {
      const unifier = new Ctor(shape, true, true)
      if (typeof unifier.Build === 'function') unifier.Build()
      else if (typeof unifier.Build_1 === 'function') unifier.Build_1()
      const result = unifier.Shape?.() ?? unifier.Shape_1?.()
      if (result) return result
    } catch { /* next */ }
    try {
      const unifier = new Ctor(shape)
      if (typeof unifier.Build === 'function') unifier.Build()
      else if (typeof unifier.Build_1 === 'function') unifier.Build_1()
      const result = unifier.Shape?.() ?? unifier.Shape_1?.()
      if (result) return result
    } catch { /* next */ }
  }

  const sewNames = Object.keys(oc).filter(k => matchExact(k, 'BRepBuilderAPI_Sewing')).sort()
  for (const n of sewNames) {
    const Ctor = oc[n]
    if (typeof Ctor !== 'function') continue
    try {
      const sewer = new Ctor(1e-6)
      const addCandidates = ['Add', 'Add_1', 'Add_2']
      for (const m of addCandidates) {
        if (typeof sewer[m] === 'function') {
          try { sewer[m](shape); break } catch { /* next */ }
        }
      }
      if (typeof sewer.Perform === 'function') sewer.Perform()
      else if (typeof sewer.Perform_1 === 'function') sewer.Perform_1()
      const result = sewer.SewedShape?.() ?? sewer.SewedShape_1?.()
      if (result) return result
    } catch { /* next */ }
  }

  return shape
}

export async function exportSTEP(oc: any, shape: any, filename = 'fluid3d.step'): Promise<Blob> {
  if (!oc?.STEPControl_Writer_1) {
    throw new Error('STEPControl_Writer_1 not found')
  }

  ensureDir(oc, '/tmp')
  const out = `/tmp/${filename}`

  const writer = new oc.STEPControl_Writer_1()
  const mode = pickStepModelType(oc)
  const progress = makeProgress(oc)

  stepTransfer(writer, shape, mode, progress)
  writer.Write(out)

  const data = oc.FS.readFile(out) as Uint8Array
  const buffer = data.buffer.slice(
    data.byteOffset,
    data.byteOffset + data.byteLength
  ) as ArrayBuffer

  return new Blob([buffer], { type: 'application/step' })
}

export async function exportSTL(oc: any, shape: any, filename = 'fluid3d.stl'): Promise<Blob> {
  ensureDir(oc, '/tmp')
  const out = `/tmp/${filename}`

  const meshNames = Object.keys(oc)
    .filter(k => matchExact(k, 'BRepMesh_IncrementalMesh'))
    .sort()
  let meshed = false
  for (const name of meshNames) {
    const Ctor = oc[name]
    if (typeof Ctor !== 'function') continue
    const attempts = [
      () => new Ctor(shape, 0.05, false, 0.5, true),
      () => new Ctor(shape, 0.05, false, 0.5),
      () => new Ctor(shape, 0.05),
    ]
    for (const create of attempts) {
      try {
        const mesh = create()
        const progress = makeProgress(oc)
        if (typeof mesh.Perform === 'function') {
          try { mesh.Perform(progress) } catch { mesh.Perform() }
        } else if (typeof mesh.Perform_1 === 'function') {
          try { mesh.Perform_1(progress) } catch { mesh.Perform_1() }
        }
        meshed = true
        break
      } catch {
        // wrong overload, try next
      }
    }
    if (meshed) break
  }
  if (!meshed) throw new Error('BRepMesh_IncrementalMesh failed')

  const writerNames = Object.keys(oc).filter(k => matchExact(k, 'StlAPI_Writer')).sort()
  const progress = makeProgress(oc)
  let written = false
  for (const name of writerNames) {
    const Ctor = oc[name]
    if (typeof Ctor !== 'function') continue
    try {
      const writer = new Ctor()
      const writeCandidates = ['Write', 'Write_1', 'Write_2']
      for (const method of writeCandidates) {
        if (typeof writer[method] !== 'function') continue
        try {
          const status = progress
            ? writer[method](shape, out, progress)
            : writer[method](shape, out)
          if (status !== false) {
            written = true
            break
          }
        } catch {
          try {
            const status = writer[method](shape, out)
            if (status !== false) {
              written = true
              break
            }
          } catch {
            // wrong overload, try next
          }
        }
      }
    } catch {
      // wrong constructor, try next
    }
    if (written) break
  }
  if (!written) throw new Error('StlAPI_Writer failed')

  const data = oc.FS.readFile(out) as Uint8Array
  const buffer = data.buffer.slice(
    data.byteOffset,
    data.byteOffset + data.byteLength,
  ) as ArrayBuffer

  return new Blob([buffer], { type: 'model/stl' })
}
