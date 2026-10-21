// Render module for the rl-tools ui-server: many airplanes at once, each flying its own test scenario (like the
// parallel environments of Isaac Lab). Sent by examples/ui_fleet.py. An airplane whose flight ended turns red.
import * as THREE from "three"
import {OrbitControls} from "three-orbitcontrols"

const SCALE = 14 // airplanes drawn larger than real so that they show from far away
const TRAIL_LENGTH = 2000
const TRAIL_SPACING = 2 // m
const ENDED_COLOR = 0xd62728

function box(size, color){
    return new THREE.Mesh(new THREE.BoxGeometry(...size), new THREE.MeshLambertMaterial({color}))
}

// Body frame: x forward, y left, z up
function airplane(b, c, color){
    const group = new THREE.Group()
    const tail_x = -0.38 * b
    const fuselage = box([0.6 * b, 0.05 * b, 0.05 * b], color)
    fuselage.position.set(-0.1 * b, 0, 0)
    const wing = box([c, b, 0.1 * c], color)
    const tail = box([0.6 * c, 0.3 * b, 0.06 * c], color)
    tail.position.set(tail_x, 0, 0)
    const fin = box([0.6 * c, 0.05 * c, 0.12 * b], color)
    fin.position.set(tail_x, 0, 0.06 * b)
    for(const part of [fuselage, wing, tail, fin]){
        group.add(part)
    }
    group.scale.setScalar(SCALE)
    return group
}

class Trail{
    constructor(color){
        this.positions = new Float32Array(TRAIL_LENGTH * 3)
        this.count = 0
        this.geometry = new THREE.BufferGeometry()
        this.geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3))
        this.line = new THREE.Line(this.geometry, new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.5}))
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
    const camera = new THREE.PerspectiveCamera(45, 1, 1, 20000)
    camera.up.set(0, 0, 1)
    camera.position.set(-750, -850, 480)
    const renderer = new THREE.WebGLRenderer({canvas, antialias: true})
    renderer.setPixelRatio(1)
    scene.add(new THREE.AmbientLight(0xffffff, 1.5))
    const sun = new THREE.DirectionalLight(0xffffff, 2)
    sun.position.set(0.3, 0.5, 1)
    scene.add(sun)

    const grid = new THREE.GridHelper(2400, 48, 0x8a96a3, 0xc3ccd6)
    grid.rotation.x = Math.PI / 2 // the helper is in the xz plane, the ground is the xy plane
    scene.add(grid)

    const controls = new OrbitControls(camera, canvas)
    controls.target.set(0, 0, 80)
    controls.update()

    const hud = document.createElement("pre")
    hud.style.cssText = "position: fixed; top: 8px; left: 12px; margin: 0; font: 15px/1.5 monospace; color: #1f2933; pointer-events: none;"
    document.body.appendChild(hud)

    return {canvas, scene, camera, renderer, controls, hud, episode: null, cursor_grab: true}
}

export async function episode_init(ui_state, parameters){
    if(ui_state.episode){
        ui_state.scene.remove(ui_state.episode)
    }
    ui_state.episode = new THREE.Group()
    ui_state.scene.add(ui_state.episode)
    ui_state.planes = []
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
    const fleet = state.fleet
    const color = state.color
    while(ui_state.planes.length < fleet.length){
        const plane = {
            group: airplane(parameters.dynamics.wing_span, parameters.dynamics.mean_chord, color),
            trail: new Trail(color),
            ended: false,
        }
        ui_state.episode.add(plane.group)
        ui_state.episode.add(plane.trail.line)
        ui_state.planes.push(plane)
    }
    let flying = 0
    fleet.forEach((s, i) => {
        const plane = ui_state.planes[i]
        plane.group.position.set(...s.position)
        plane.group.quaternion.set(s.orientation[1], s.orientation[2], s.orientation[3], s.orientation[0]).normalize()
        plane.trail.push(s.position)
        if(s.ended && !plane.ended){
            plane.ended = true
            plane.group.traverse((part) => { if(part.material){ part.material.color.setHex(ENDED_COLOR) } })
            plane.trail.line.material.color.setHex(ENDED_COLOR)
        }
        flying += plane.ended ? 0 : 1
    })
    ui_state.controls.update()
    ui_state.hud.textContent = [
        state.title,
        `${fleet.length} test scenarios at once, t = ${state.t.toFixed(1)} s`,
        `flying: ${flying}   ended (red): ${fleet.length - flying}`,
    ].join("\n")
    ui_state.renderer.render(ui_state.scene, ui_state.camera)
}
