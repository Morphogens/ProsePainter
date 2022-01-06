// A file to store the history of the edits to a canvas.
import { get } from 'svelte/store'
import { DEFAULT_IMAGE } from './constants'
import { canvasToImage, loadImage } from './utils'
import { mainCanvas, maskCanvas, prompt, lastOptimizationResult, localStorageWritable } from './stores'
import { events } from './optimizeEvents'

interface DrawHistory {
    startImage: string
    hasOptimized: boolean
    // TODO, store masks and prompts
}

export const drawHistoryStore = localStorageWritable<DrawHistory>('drawHistory', {
    startImage: '',
    hasOptimized: false
})

// let startImage: null | string = null
// let data = []

events.addEventListener('accepted', () => {
    console.log('Accepted.')
    // data.push({
    //     prompt: get(prompt),
    //     mask: canvasToImage(get(maskCanvas).getCanvas())
    // })
    drawHistoryStore.update((_drawHistoryStore) => ({
        ..._drawHistoryStore,
        hasOptimized: true,
    }))
})

export function reset() {
    // Called when draw canvas is updated.
    console.log('draw history reset')
    drawHistoryStore.update((_drawHistoryStore) => ({
        hasOptimized: false,
        startImage: get(mainCanvas).getCanvas().toDataURL('image/jpeg', 90)
    }))
}

export async function exportImage(): Promise<HTMLCanvasElement> {
    const finalCanvas = get(mainCanvas).getCanvas()
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    canvas.width = (finalCanvas.width * 2)
    canvas.height = finalCanvas.height
    // The start image will be an empty string on the first load.
    const startImageUrl = get(drawHistoryStore).startImage || DEFAULT_IMAGE
    const startImage = await loadImage(startImageUrl)
    ctx.drawImage(startImage, 0, 0)
    ctx.drawImage(finalCanvas, finalCanvas.width, 0)
    return canvas
}