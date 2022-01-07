import { Mode } from './types';
import { writable, get } from 'svelte/store'
import { addEventListener, messageServer } from "@/lib/socket";
import { prompt, mode, learningRate, optimizationResults, mainCanvas, maskCanvas, stylePrompt, numRecSteps, modelType, numUsers, selectedOptIdx, selectedOptImage } from './stores'
import { imgTob64 } from './utils';

interface StartGenerationData {
    prompt: string
    stylePrompt: string,
    imageBase64: string,
    learningRate: number,
    backgroundImg: string,
    numRecSteps: number | null,
    modelType: string
}

function getGenerationData(): StartGenerationData {
    return {
        prompt: get(prompt),
        stylePrompt: get(stylePrompt) ?? '',
        learningRate: get(learningRate) / 1000,
        imageBase64: get(maskCanvas).canvasBase64,
        backgroundImg: get(mainCanvas).canvasBase64,
        numRecSteps: get(numRecSteps),
        modelType: get(modelType)
    }
}

function validateGenerationData(data: StartGenerationData) {
    for (const [key, value] of Object.entries(data)) {
        if (key == 'stylePrompt') {
            continue
        }
        if (key == 'numRecSteps') {
            continue
        }
        if (!value || value.length == 0) {
            throw new Error(`Empty value for: ${key}.`)
        }
    }
}

function lastOptimizationResult():HTMLImageElement {
    const images = get(optimizationResults).images
    return images[images.length]
}

export const events: EventTarget = new EventTarget()

export function start() {
    const data = getGenerationData()
    validateGenerationData(data)
    messageServer('start-generation', data)
    mode.set(Mode.Optimizing)
}

export function discard() {
    messageServer("stop-generation", {})
    optimizationResults.set(null)
    mode.set(Mode.MaskDraw)
}

export function accept() {
    messageServer("stop-generation", {})
    get(mainCanvas).setImage(get(selectedOptImage))
    optimizationResults.set(null)
    mode.set(Mode.MaskDraw)
    events.dispatchEvent(new CustomEvent('accepted', {}))
}

// export function upscale() {
//     const data = getGenerationData()
//     data.backgroundImg = imgTob64(get(lastOptimizationResult).image)
//     validateGenerationData(data)
//     messageServer('upscale-generation', data)
//     mode.set(Mode.Optimizing)
// }

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
    const data = getGenerationData()
    data.backgroundImg = imgTob64(lastOptimizationResult())
    validateGenerationData(data)
    messageServer('resume-generation', data)
    mode.set(Mode.Optimizing)
}

addEventListener("message", (e) => {
    console.log("MESSAGE RECEIVED!")
    const message = JSON.parse(e.data)
    if (message.numUsers) {
        numUsers.set(message.numUsers);
    }
    if (get(mode) !== Mode.Optimizing) {
        // If we get a message from the server after the user paused.
        return
    }
    if (message.image) {
        console.log("IMAGE RECEIVED!")
        const newImage = new Image()
        newImage.src = "data:text/plain;base64," + message.image
        optimizationResults.update(_optResults => ({
            images: (_optResults ? _optResults.images : []).concat([newImage]),
            num_iterations: message.num_iterations as number,
        }))
        if (message.step == message.num_iterations) {
            mode.set(Mode.PausedOptimizing)
        }
        selectedOptIdx.set(message.step - 1)
    } else {
        console.log("NO IMAGE RECEIVED!")
    }
});