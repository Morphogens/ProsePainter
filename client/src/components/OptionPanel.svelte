<script lang="ts">
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import { mode } from "@/stores";
    import * as optEvents from "../optimizeEvents";
    import { Mode } from "../types";
    import DrawMaskOptions from "./DrawMaskOptions.svelte";
    import DrawDirectOptions from "./DrawDirectOptions.svelte";
import SetImageOptions from "./SetImageOptions.svelte";
    export let maskCanvas: DrawCanvas;
    export let mainCanvas: DrawCanvas;
    $: console.log('mode=', $mode);
    
</script>

{#if $mode != Mode.Optimizing && $mode != Mode.PausedOptimizing}
    <div id="modeToggle">
        <button
            class:selected={$mode == Mode.SetImage}
            on:click={(e) => ($mode = Mode.SetImage)}><p>Set Image</p></button
        >
        <button
            class:selected={$mode == Mode.DirectDraw}
            on:click={(e) => ($mode = Mode.DirectDraw)}><p>Sketch</p></button
        >
        <button
            class:selected={$mode == Mode.MaskDraw}
            on:click={(e) => ($mode = Mode.MaskDraw)}><p>Magic Draw</p></button
        >
    </div>
{/if}
<div id="optionPanel">
    <SetImageOptions {mainCanvas} />
    <DrawMaskOptions {maskCanvas} />
    <DrawDirectOptions {mainCanvas} />

    {#if $mode == Mode.Optimizing}
        <button on:click={() => optEvents.pause()}>
            <h4>Stop</h4>
        </button>
    {:else if $mode == Mode.PausedOptimizing}
        <button on:click={() => optEvents.accept()}>
            <h4>Accept</h4>
        </button>
        <button on:click={() => optEvents.discard()}>
            <h4>Discard</h4>
        </button>
        <button on:click={() => optEvents.upscale()}>
            <h4>Upscale</h4>
        </button>
        <button on:click={() => optEvents.resume()}>
            <h4>Resume</h4>
        </button>
    {/if}
    <!-- {/if} -->
</div>

<style>
    #optionPanel {
        position: fixed;
        left: 0px;
        top: 60px;
        z-index: 2;
        width: 150px;
        display: flex;
        flex-direction: column;
        background: white;
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
        align-items: center;
        border: 1px solid;
    }
    #modeToggle {
        position: fixed;
        top: 0px;
        left: 0px;
        background: white;
        z-index: 1;
    }
    :global(button > img) {
        width: 50px;
        height: 50px;
    }
    :global(button.selected) {
        background: #ffa50080;
    }
    :global(input) {
        width: 96px;
    }
    /* :global(p) {
        text-align: center;
        margin-bottom: 2px;
    } */
    :global(input[type="range"][orient="vertical"]) {
        writing-mode: bt-lr; /* IE */
        -webkit-appearance: slider-vertical; /* WebKit */
        width: 8px;
        height: 100px;
        padding: 0 5px;
    }
    :global(#optionPanel p ) {
        text-align: center;
        /* margin-bottom: 2px; */
        margin:4px;
        margin-top: 8px;
    }
</style>
