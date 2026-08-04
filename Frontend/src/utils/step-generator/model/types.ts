export interface Point3 { x: number; y: number; z: number }

export interface CircleShape {
    type: 'Circle'
    center: Point3
    radius: number
    height: number
    fill?: boolean
}

export interface LineShape {
    type: 'Line'
    start: Point3
    end:   Point3
    width: number
    height: number
    fill?: boolean
}

export interface PolygonShape {
    type: 'Polygon'
    points : Point3[]
    direction: Point3
    fill?: boolean
}

export interface CurveShape {
    type: 'Curve'
    start: Point3
    end: Point3
    center: Point3
    tangent?: Point3
    width: number
    height: number
    fill?: boolean
}

export interface ChamferShape {
    type: 'Chamfer'
    center: Point3
    radius: number
    radius_top: number
    height: number
    fill?: boolean
}

export type Shape = CircleShape | LineShape | PolygonShape | CurveShape | ChamferShape

export interface Channel {
    shapes: Shape[]
}

export interface Component {
    id: string
    shapes: Shape[]
    channels: Channel[]
}

export interface CompensationPolygon {
    type: 'Polygon'
    points : Point3[]
    direction: Point3
    fill?: boolean
}
export interface Layer {
    elevation: number
    components: Component[]
    channels: Channel[]
    compensation?: CompensationPolygon
}

export interface CrossLayerConnection {
    shapes: Shape[]
}

export interface ChipJSON {
    layers: Layer[]
    crosslayerConnections: CrossLayerConnection[]
    relation?: any[]
    general: {
        length: number
        width: number
        thickness: number
        precision: 'High' | 'Medium' | 'Low'
    }
}
