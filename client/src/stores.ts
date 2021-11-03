import { writable,readable, derived, get } from 'svelte/store'
import { socket, messageServer } from "@/lib/socket";
import startBackgroundUrl from './assets/startImage0.jpeg'
import { loadImage, imgTob64 } from './utils';
import { canvasBase64 } from './drawing/stores'

export const isOptimizing = writable(false)
export const lastOptimizationResult = writable(new Image(512, 512))
export const prompt = writable('a dog')
export const learningRate = writable(250)
loadImage(startBackgroundUrl).then(img => lastOptimizationResult.set(img))

interface StartGenerationData {
    prompt: string
    imageBase64: string
    learningRate: number
    backgroundImg: string
}

export function startGeneration() {
    const data: StartGenerationData = {
        prompt: get(prompt),
        imageBase64: get(canvasBase64),
        learningRate: get(learningRate)/1000,
        backgroundImg: imgTob64(get(lastOptimizationResult)),
    }
    if (data.prompt == '') {
        return console.warn('Need a promp to optimize.')
    } 
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