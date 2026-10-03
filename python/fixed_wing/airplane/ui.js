// Render module for the rl-tools ui-server (counterpart of get_ui in l2f/operations_cpu.h).
// The generic client imports this file and calls init, episode_init and render.
import * as THREE from "three"
import {OrbitControls} from "three-orbitcontrols"

const TRAIL_LENGTH = 5000
const TRAIL_SPACING = 0.5 // m
const GRID_CELL = 10 // m
const GRID_CELLS = 200

function box(size, color){
    return new THREE.Mesh(new THREE.BoxGeometry(...size), new THREE.MeshLambertMaterial({color}))
}

// A control surface hinged at its leading edge, extending backwards (-x)
function control_surface(size, hinge_position, color){
    const hinge = new THREE.Group()
    const surface = box(size, color)
    surface.position.set(-size[0] / 2, 0, 0)
    hinge.add(surface)
    hinge.position.set(...hinge_position)
    return hinge
}

// Body frame: x forward, y left, z up
class Airplane{
    constructor(parameters){
        const b = parameters.dynamics.wing_span
        const c = parameters.dynamics.mean_chord
        const body_color = 0x4b5563
        const surface_color = 0xe8743b
        const tail_x = -0.38 * b
        const hinge_x = tail_x - 0.3 * c
        this.group = new THREE.Group()

        const fuselage = box([0.6 * b, 0.05 * b, 0.05 * b], body_color)
        fuselage.position.set(-0.1 * b, 0, 0)
        const wing = box([c, b, 0.1 * c], body_color)
        const horizontal_tail = box([0.6 * c, 0.3 * b, 0.06 * c], body_color)
        horizontal_tail.position.set(tail_x, 0, 0)
        const fin = box([0.6 * c, 0.05 * c, 0.12 * b], body_color)
        fin.position.set(tail_x, 0, 0.06 * b)

        this.aileron_left = control_surface([0.3 * c, 0.3 * b, 0.08 * c], [-0.5 * c, 0.35 * b, 0], surface_color)
        this.aileron_right = control_surface([0.3 * c, 0.3 * b, 0.08 * c], [-0.5 * c, -0.35 * b, 0], surface_color)
        this.elevator = control_surface([0.4 * c, 0.3 * b, 0.05 * c], [hinge_x, 0, 0], surface_color)
        this.rudder = control_surface([0.4 * c, 0.04 * c, 0.12 * b], [hinge_x, 0, 0.06 * b], surface_color)

        this.propeller = new THREE.Mesh(
            new THREE.CylinderGeometry(0.07 * b, 0.07 * b, 0.01 * b, 32),
            new THREE.MeshLambertMaterial({color: 0x2a6fdb, transparent: true})
        )
        this.propeller.rotation.set(0, 0, Math.PI / 2)
        this.propeller.position.set(0.21 * b, 0, 0)

        for(const part of [fuselage, wing, horizontal_tail, fin, this.aileron_left, this.aileron_right, this.elevator, this.rudder, this.propeller]){
            this.group.add(part)
        }
    }
    set_actuators(actuators){
        // aerospace sign convention: positive aileron rolls right, positive elevator pitches down (trailing edge down),
        // positive rudder yaws left (trailing edge left)
        const [aileron, elevator, rudder, throttle] = actuators
        this.aileron_left.rotation.y = -aileron
        this.aileron_right.rotation.y = aileron
        this.elevator.rotation.y = -elevator
        this.rudder.rotation.z = -rudder
        this.propeller.material.opacity = 0.1 + 0.8 * throttle
    }
}

class Trail{
    constructor(){
        this.positions = new Float32Array(TRAIL_LENGTH * 3)
        this.count = 0
        this.geometry = new THREE.BufferGeometry()
        this.geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3))
        this.line = new THREE.Line(this.geometry, new THREE.LineBasicMaterial({color: 0x2a6fdb}))
        this.line.frustumCulled = false
        this.last = null
    }
    push(position){
        if(this.last && Math.hypot(...position.map((p, i) => p - this.last[i])) < TRAIL_SPACING){
            return
        }
        if(this.count === TRAIL_LENGTH){
            this.positions.copyWithin(0, 3)
            this.count -= 1
        }
        this.positions.set(position, this.count * 3)
        this.count += 1
        this.last = position
        this.geometry.setDrawRange(0, this.count)
        this.geometry.attributes.position.needsUpdate = true
    }
}

export async function init(canvas, options){
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0xeef3f8)
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000)
    camera.up.set(0, 0, 1)
    const renderer = new THREE.WebGLRenderer({canvas, antialias: true})
    renderer.setPixelRatio(1)
    scene.add(new THREE.AmbientLight(0xffffff, 1.5))
    const sun = new THREE.DirectionalLight(0xffffff, 2)
    sun.position.set(0.3, 0.5, 1)
    scene.add(sun)

    const grid = new THREE.GridHelper(GRID_CELL * GRID_CELLS, GRID_CELLS, 0x8a96a3, 0xc3ccd6)
    grid.rotation.x = Math.PI / 2 // the helper is in the xz plane, the ground is the xy plane
    scene.add(grid)

    const controls = new OrbitControls(camera, canvas)

    const hud = document.createElement("pre")
    hud.style.cssText = "position: fixed; top: 8px; left: 12px; margin: 0; font: 13px/1.5 monospace; color: #1f2933; pointer-events: none;"
    document.body.appendChild(hud)

    return {canvas, scene, camera, renderer, controls, grid, hud, episode: null, cursor_grab: true}
}

export async function episode_init(ui_state, parameters){
    if(ui_state.episode){
        ui_state.scene.remove(ui_state.episode)
    }
    const episode = new THREE.Group()
    ui_state.episode = episode
    ui_state.scene.add(episode)

    ui_state.airplane = new Airplane(parameters)
    episode.add(ui_state.airplane.group)
    ui_state.trail = new Trail()
    episode.add(ui_state.trail.line)

    // vertical line down to the ground, to read the altitude
    ui_state.altitude_geometry = new THREE.BufferGeometry()
    ui_state.altitude_geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3))
    const altitude_line = new THREE.Line(ui_state.altitude_geometry, new THREE.LineBasicMaterial({color: 0x8a96a3}))
    altitude_line.frustumCulled = false
    episode.add(altitude_line)

    const wind = parameters.disturbances.wind
    const wind_speed = Math.hypot(...wind)
    ui_state.wind_arrow = null
    if(wind_speed > 0){
        const b = parameters.dynamics.wing_span
        ui_state.wind_arrow = new THREE.ArrowHelper(new THREE.Vector3(...wind).normalize(), new THREE.Vector3(), 0.15 * b * wind_speed, 0x1a9e6b, 0.15 * b, 0.08 * b)
        episode.add(ui_state.wind_arrow)
    }
    ui_state.last_position = null
}

function update_size(ui_state){
    const {canvas, renderer, camera} = ui_state
    if(canvas.width !== ui_state.width || canvas.height !== ui_state.height){
        ui_state.width = canvas.width
        ui_state.height = canvas.height
        renderer.setSize(canvas.width, canvas.height, false)
        camera.aspect = canvas.width / canvas.height
        camera.updateProjectionMatrix()
    }
}

export async function render(ui_state, parameters, state, action){
    update_size(ui_state)
    const b = parameters.dynamics.wing_span
    const position = new THREE.Vector3(...state.position)
    const airplane = ui_state.airplane
    airplane.group.position.copy(position)
    airplane.group.quaternion.set(state.orientation[1], state.orientation[2], state.orientation[3], state.orientation[0]).normalize()
    airplane.set_actuators(state.actuators)
    ui_state.trail.push(state.position)

    const altitude = ui_state.altitude_geometry.attributes.position
    altitude.setXYZ(0, position.x, position.y, position.z)
    altitude.setXYZ(1, position.x, position.y, 0)
    altitude.needsUpdate = true
    if(ui_state.wind_arrow){
        ui_state.wind_arrow.position.set(position.x, position.y, position.z + 0.4 * b)
    }
    ui_state.grid.position.set(Math.round(position.x / GRID_CELL) * GRID_CELL, Math.round(position.y / GRID_CELL) * GRID_CELL, 0)

    // chase camera: the camera moves with the airplane, the mouse orbits around it
    if(ui_state.last_position){
        ui_state.camera.position.add(position.clone().sub(ui_state.last_position))
    }
    else{
        ui_state.camera.position.copy(position).add(new THREE.Vector3(-2 * b, -2 * b, 1.2 * b))
    }
    ui_state.last_position = position
    ui_state.controls.target.copy(position)
    ui_state.controls.update()

    const deg = (rad) => (rad * 180 / Math.PI).toFixed(1).padStart(6)
    const num = (x) => x.toFixed(1).padStart(6)
    const wind = parameters.disturbances.wind
    ui_state.hud.textContent = [
        `airspeed     ${num(state.air_data.airspeed)} m/s`,
        `ground speed ${num(Math.hypot(...state.linear_velocity))} m/s`,
        `climb rate   ${num(state.linear_velocity[2])} m/s`,
        `altitude     ${num(state.position[2])} m`,
        `alpha        ${deg(state.air_data.alpha)} deg`,
        `beta         ${deg(state.air_data.beta)} deg`,
        `energy       ${num(state.specific_energy / 9.81)} m`,
        `wind         ${num(Math.hypot(...wind))} m/s`,
        ``,
        `aileron      ${deg(state.actuators[0])} deg`,
        `elevator     ${deg(state.actuators[1])} deg`,
        `rudder       ${deg(state.actuators[2])} deg`,
        `throttle     ${num(state.actuators[3] * 100)} %`,
    ].join("\n")

    ui_state.renderer.render(ui_state.scene, ui_state.camera)
}
