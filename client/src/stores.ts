import { writable,readable, derived, get } from 'svelte/store'
// import type { DrawLayer } from './types'
// import type { Readable } from 'svelte/store'
import { socket, socketOpen, messageServer } from "@/lib/socket";
import startBackgroundUrl from './assets/startImage0.jpeg'
import { loadImage, imgTob64 } from './utils';
import { canvasBase64 } from './drawing/stores'

export const isOptimizing = writable(false)
export const lastOptimizationResult = writable(new Image(512, 512))
export const prompt = writable('a dog')

loadImage(startBackgroundUrl).then(img => lastOptimizationResult.set(img))

export function startGeneration() {
    const data = {
        prompt: get(prompt),
        imageBase64: canvasBase64,
        backgroundImg: imgTob64(get(lastOptimizationResult)),
    }
    if (data.prompt == '') {
        return console.warn('Need a promp to optimize.')
    } 
    console.log(data);

    messageServer('start-generation', data)
    isOptimizing.set(true)
}

export function stopGeneration(){
    messageServer("stop-generation", {})
    isOptimizing.set(false)
}

socket.addEventListener("message", (e) => {
    console.log("MESSAGE RECEIVED!")
    const message = JSON.parse(e.data)
    if (message.image) {
        console.log("IMAGE RECEIVED!")
        const newImage = new Image()
        newImage.src = "data:text/plain;base64," + message.image
        lastOptimizationResult.set(newImage)
    } else {
        console.log("NO IMAGE RECEIVED!")
    }
});