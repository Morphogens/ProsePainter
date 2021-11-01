import { writable, get } from 'svelte/store'

export const radius = writable(16)
export const opacity = writable(0.0)
export const softness = writable(3)
export const erasing = writable(false)

interface DrawContext {
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement
}
export const drawContext = writable(null as null | DrawContext)
export const canvasBase64 = writable(null as null | string)
const history:string[] = []

canvasBase64.subscribe(() => {
    console.log('canvasBase64 changed')
})

export function canvasChanged() {
    const { canvas } = get(drawContext)
    const base64 = canvas.toDataURL()
    history.push(get(canvasBase64))
    canvasBase64.set(base64)
    window.localStorage.setItem('canvasBase64', base64)
}

export function clear() {
    const { canvas, ctx } = get(drawContext)
    ctx.clearRect(0, 0, 512, 512)
    canvasChanged()
}

export function undo() {
    if (history.length) {
        canvasBase64.set(history.pop())
    }
}
// export function saveState(canvas:HTMLCanvasElement) {
//     const newBase64 = canvas.toDataURL()
//     history.push(get(canvasBase64))
//     canvasBase64.set(newBase64)
//     // console.log('saving', newBase64.length);
//     window.localStorage.setItem('canvasBase64', newBase64)
// }
