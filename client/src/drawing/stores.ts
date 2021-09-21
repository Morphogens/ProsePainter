import { writable, derived, get } from 'svelte/store'
import color from 'randomcolor'
import type { DrawLayer } from './types'

export const radius = writable(16)
export const erasing = writable(false)

export const layers = writable([] as DrawLayer[])
export const activeLayerIdx = writable(null as null | number)
export const activeLayer = derived([activeLayerIdx, layers], ([$activeLayerIdx, $layers]) => {
    return $activeLayerIdx != null ? $layers[$activeLayerIdx] : null
})

export function addLayer(prompt:string) {
    const layer = {
        prompt,
        alpha: 1.0,
        data: null,
        color: color()
    }    
    console.log(layer.color);
    
    layers.update($layers => [...$layers, layer])
    activeLayerIdx.set(get(layers).length - 1)
}