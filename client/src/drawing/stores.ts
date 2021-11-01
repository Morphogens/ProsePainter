import { writable,readable, derived, get } from 'svelte/store'
import type { Readable } from 'svelte/store'

export const radius = writable(16)
export const opacity = writable(0.0)
export const softness = writable(3)
export const erasing = writable(false)

export const canvasBase64 = writable(null as null | string)

const history:string[] = []

// export function addToHistory(canvas:HTMLCanvasElement()) [
// ]

canvasBase64.subscribe(() => {
    console.log('canvasBase64 changed')
})

export function undo() {
    if (history.length) {
        canvasBase64.set(history.pop())
    }
}

export function saveState(canvas:HTMLCanvasElement) {
    const newBase64 = canvas.toDataURL()
    history.push(get(canvasBase64))
    canvasBase64.set(newBase64)
    // console.log('saving', newBase64.length);

    window.localStorage.setItem('canvasBase64', newBase64)
}

export function clear() {

}