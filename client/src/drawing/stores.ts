import { writable, derived, get } from 'svelte/store'
import type { Readable } from 'svelte/store'
import * as Y from 'yjs'
import { IndexeddbPersistence } from 'y-indexeddb'
import { readableArray, readableMap } from 'svelt-yjs'

import color from 'randomcolor'
import type { DrawLayer } from './types'

const ydoc = new Y.Doc()
new IndexeddbPersistence('clipCanvas', ydoc)


export const radius = writable(16)
export const erasing = writable(false)

const yLayers = ydoc.get(`ylayers`, Y.Array) as Y.Array<Y.Map<any>>
export const undo = new Y.UndoManager(yLayers)

export const layers = readableArray(yLayers)
export const activeLayerIdx = writable(null as null | number)
export const activeLayer = derived([activeLayerIdx, layers], ([$activeLayerIdx, $layers]) => {
    return $activeLayerIdx != null ? $layers[$activeLayerIdx] : null
})
export const layerImages = new Map<Y.Map<any>, HTMLImageElement>()

export function addLayer(prompt:string) {
    const yMap = new Y.Map<any>()
    // const layer:DrawLayer = {
        //     prompt,
        //     color: color(),
        //     strength: 1.0,
        //     image: null,
        //     imageBase64: null,
        // }    
    yMap.set('prompt', prompt)
    yMap.set('color', color())
    yMap.set('strength', 1.0)
    yMap.set('imageBase64', null)
    yLayers.insert(yLayers.length, [yMap])
    activeLayerIdx.set(get(layers).length - 1)
}

layers.subscribe($layers => {
    $layers.forEach(bindYLayer)
})

function bindYLayer(yMap:Y.Map) {
    if (layerImages.get(yMap)) {
        return
    }
    const image = new Image()
    layerImages.set(yMap, image)    
    if (yMap.get('imageBase64')) {
        image.src = yMap.get('imageBase64')
    }
    yMap.observe(() => {
        if (yMap.get('imageBase64')) {
            // console.log('iage change');
            
            image.src = yMap.get('imageBase64')
        }
    })
}
