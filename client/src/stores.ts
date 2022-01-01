import type { OptimizationResult } from './types'
import { Mode } from './types'
import { writable } from 'svelte/store'
import type { Writable } from 'svelte/store'
import type DrawCanvas from './components/DrawCanvas'

export const prompt = localStorageWritable('prompt', 'a dog')
export const stylePrompt = localStorageWritable('stylePrompt', '')
export const learningRate = localStorageWritable('learningRate', 250)
export const numRecSteps = localStorageWritable('numRecSteps', 0)
export const modelType = localStorageWritable("modelType", "imagenet-16384")
export const mode = writable(Mode.MaskDraw as Mode)
export const canvasSize = localStorageWritable('canvasSize', [512, 512])
export const mainCanvas = writable(null as null | DrawCanvas)
export const maskCanvas = writable(null as null | DrawCanvas)


export const lastOptimizationResult = writable(null as null | OptimizationResult)

function localStorageWritable<T>(name:string, defaultValue:T):Writable<T>{
    // Svelte store that persists to localStorage.
    // Data must be JSON-able.
    const store = writable(
        JSON.parse(localStorage.getItem(name)) ?? defaultValue
    )
    store.subscribe($value => {
        localStorage.setItem(name, JSON.stringify($value))
    })
    return store
}