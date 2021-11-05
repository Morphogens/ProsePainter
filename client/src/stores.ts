import { Mode } from './types'
import { writable } from 'svelte/store'
import type DrawCanvas from './components/DrawCanvas'
export const prompt = writable('a dog')
export const stylePrompt = writable('')
export const lastOptimizationResult = writable(null as null | HTMLImageElement)
export const learningRate = writable(250)
export const mode = writable(Mode.MaskDraw as Mode)
export const maskCanvasBase64 = writable(null as null | string)
export const mainCanvasBase64 = writable(null as null | string)
export const mainCanvas = writable(null as null | DrawCanvas)
export const maskCanvas = writable(null as null | DrawCanvas)
export const canvasSize = writable([512, 512] as [number, number])