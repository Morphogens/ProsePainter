<script lang="ts">
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import { learningRate, prompt, mode, canvasSize, stylePrompt, numRecSteps } from "@/stores";
    import * as optEvents from "../optimizeEvents";
    import { socketOpen } from "@/lib/socket";
    import { Mode } from "../types";
    import { loadImage } from "@/utils";
    import { tick } from "svelte";
    export let maskCanvas: DrawCanvas;
    export let mainCanvas: DrawCanvas;

    async function onFiles(event) {
        const { files } = event.target;
        if (files.length) {
            const url = URL.createObjectURL(files[0]);
            const image = await loadImage(url);
            canvasSize.set([ image.width, image.height ])
            await tick() // DrawCanvases get recreated.
            mainCanvas.set(image)
        }
    }
</script>

<div id="optionPanel">
    <!-- <button>

    </button> -->
    {#if $mode == Mode.MaskDraw && maskCanvas}
        <p>What would you like to draw?</p>
        <input type="text" minlength="3" bind:value={$prompt} />
        <p>In what style?</p>
        <input type="text" placeholder="Default" bind:value={$stylePrompt} />
        <br>
        <button
            on:click={() => (maskCanvas.erasing = false)}
            class:selected={!maskCanvas.erasing}
        >
            <img src="/pencil.svg" alt="draw-mask" />
        </button>
        <button
            on:click={() => (maskCanvas.erasing = true)}
            class:selected={maskCanvas.erasing}
        >
            <img src="/eraser.png" alt="erase-mask" />
        </button>
        <br>

        <button on:click={() => maskCanvas.clear()}>
            <p>Clear Mask</p>
        </button>

        <p>Radius</p>
        <input type="range" bind:value={maskCanvas.radius} min="1" max="96" />
        <p>{maskCanvas.radius}</p>

        <p>Softness</p>
        <input
            type="range"
            bind:value={maskCanvas.softness}
            min="0"
            max="20"
            step="any"
        />
        {maskCanvas.softness}

        <p>Learning Rate</p>
        <input
            type="range"
            bind:value={$learningRate}
            min="0"
            max="500.0"
            step="1"
        />
        {$learningRate / 1000}
        
        <p>Reconstruction Steps</p>
        <input
            type="range"
            bind:value={$numRecSteps}
            min="0"
            max="64"
            step=1
        />
        {$numRecSteps}

        {#if $socketOpen}
            <button on:click={() => optEvents.start()} style='background-color: #4caf50;'>
                <h4>Start</h4>
            </button>
        {:else}
            <button on:click={() => alert("SOCKET NOT CONNECTED :(")}>
                Start
            </button>
            <p>Socket not open</p>
        {/if}
    {:else if $mode == Mode.DirectDraw && mainCanvas}
        <input
            type="color"
            id="head"
            name="head"
            bind:value={mainCanvas.strokeColor}
        />
        <label for="head">Head</label>
        <p>Radius={mainCanvas.radius}</p>
        <input type="range" bind:value={mainCanvas.radius} min="1" max="96" />
        <p>Softness</p>
        <input
            type="range"
            bind:value={mainCanvas.softness}
            min="0"
            max="20"
            step="any"
        />
        <p>Select an image</p>
        <input type="file" accept=".jpg, .jpeg, .png" on:change={onFiles} />
    {:else if $mode == Mode.Optimizing}
        <button on:click={() => optEvents.pause()}>
            <h4>Stop</h4>
        </button>
    {:else if $mode == Mode.PausedOptimizing}
        <button on:click={() => optEvents.accept()}>
            <h4>Accept</h4>
        </button>
        <button on:click={() => optEvents.resume()}>
            <h4>Resume</h4>
        </button>
        <button on:click={() => optEvents.discard()}>
            <h4>Discard</h4>
        </button>
    {/if}
</div>

<style>
    #optionPanel {
        position: fixed;
        left: 0px;
        top: 60px;
        z-index: 2;
        width: 128px;
        display: flex;
        flex-direction: column;
        background: white;
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
        align-items: center;
        border: 1px solid;
    }
    button > img {
        width: 50px;
        height: 50px;
    }
    button.selected {
        background: #ffa50080;
    }
    input {
        width: 96px;
    }
    p {
        text-align: center;
        margin-bottom: 2px;
    }
    input[type="range"][orient="vertical"] {
        writing-mode: bt-lr; /* IE */
        -webkit-appearance: slider-vertical; /* WebKit */
        width: 8px;
        height: 100px;
        padding: 0 5px;
    }
</style>
