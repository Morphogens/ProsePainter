import type { DrawLayer } from './types'
import { writable,readable, derived, get } from 'svelte/store'
import type { Readable } from 'svelte/store'
import { socket, socketOpen, messageServer } from "@/lib/socket";
import startBackgroundUrl from './assets/startImage0.jpeg'
import { loadImage, imgTob64 } from './utils';
export const isOptimizing = writable(false)
// export const backgroundImageBase64 = writable('')
// export const backgroundImage = new Image()
// backgroundImage.src = startBackgroundUrl
// const startImage = new Image()
// startImage.src = 

export const lastOptimizationResult = writable(new Image(512, 512))
export const maskCanvas = document.createElement('canvas') as HTMLCanvasElement
export const prompt = writable('a dog')

loadImage(startBackgroundUrl).then(img => lastOptimizationResult.set(img))

export function startGeneration() {
    const data = {
        prompt: get(prompt),
        imageBase64: maskCanvas.toDataURL(),
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