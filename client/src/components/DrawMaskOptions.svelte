<script lang="ts">
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import {
        learningRate,
        prompt,
        mode,
        stylePrompt,
    } from "@/stores";
    import * as optEvents from "../optimizeEvents";
    import { socketOpen } from "@/lib/socket";
    import { Mode } from "../types";
    export let maskCanvas: DrawCanvas;
</script>

{#if $mode == Mode.MaskDraw && maskCanvas}
    <p>What would you like to draw?</p>
    <input type="text" minlength="3" bind:value={$prompt} />
    <p>In what style?</p>
    <input type="text" placeholder="Default" bind:value={$stylePrompt} />
    <br />
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
    <br />
    <button on:click={() => maskCanvas.clear()}>
        <p>Clear Mask</p>
    </button>
    <p>Radius={maskCanvas.radius}</p>
    <input type="range" bind:value={maskCanvas.radius} min="1" max="96" />
    <p>Softness</p>
    <input
        type="range"
        bind:value={maskCanvas.softness}
        min="0"
        max="20"
        step="any"
    />
    <p>learningRate</p>
    <input
        type="range"
        bind:value={$learningRate}
        min="0"
        max="500.0"
        step="1"
    />
    {$learningRate / 1000}

    {#if $socketOpen}
        <button
            on:click={() => optEvents.start()}
            style="background-color: #4caf50;"
        >
            <h4>Start</h4>
        </button>
    {:else}
        <button on:click={() => alert("SOCKET NOT CONNECTED :(")}>
            Start
        </button>
        <p>Socket not open</p>
    {/if}
{/if}
