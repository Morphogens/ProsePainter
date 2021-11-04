import { Mode } from './types';
import { writable, get } from 'svelte/store'
import { socket, messageServer } from "@/lib/socket";
import { prompt, mode, learningRate, maskCanvasBase64, mainCanvasBase64, lastOptimizationResult, mainCanvas } from './stores'
import { imgTob64 } from './utils';

interface StartGenerationData {
    prompt: string
    imageBase64: string
    learningRate: number
    backgroundImg: string
}

export function start() {
    const data: StartGenerationData = {
        prompt: get(prompt),
        learningRate: get(learningRate),
        imageBase64: get(maskCanvasBase64),
        backgroundImg: get(mainCanvasBase64),
    }
    for (const [key, value] of Object.entries(data)) {
        if (!value || value.length == 0) {
            throw new Error(`Empty value for: ${key}.`)
        }
    }
    messageServer('start-generation', data)
    mode.set(Mode.Optimizing)
}

export function discard() {
    messageServer("stop-generation", {})
    lastOptimizationResult.set(null)
    mode.set(Mode.MaskDraw)
}

export function accept() {
    messageServer("stop-generation", {})
    get(mainCanvas).set(get(lastOptimizationResult))
    lastOptimizationResult.set(null)
    mode.set(Mode.MaskDraw)
}

export function pause() {
    // TODO
    // messageServer("pause-generation", {})
    messageServer("stop-generation", {})
    mode.set(Mode.PausedOptimizing)
}

export function resume() {
    // TODO
    // messageServer("resume-generation", {})
    // like start() but use lastOptimizationResult instead of main canvas
    const data: StartGenerationData = {
        prompt: get(prompt),
        learningRate: get(learningRate),
        imageBase64: get(maskCanvasBase64),
        backgroundImg: imgTob64(get(lastOptimizationResult)),
    }
    for (const [key, value] of Object.entries(data)) {
        if (!value || value.length == 0) {
            throw new Error(`Empty value for: ${key}.`)
        }
    }
    messageServer('start-generation', data)
    mode.set(Mode.Optimizing)
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