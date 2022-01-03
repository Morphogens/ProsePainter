<script lang="ts">
    import type DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import { mode } from "@/stores";
    import { Mode } from "../types";
    import DrawMaskOptions from "./DrawMaskOptions.svelte";
    import DrawDirectOptions from "./DrawDirectOptions.svelte";
    import OptimizeOptions from './OptimizeOptions.svelte'
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
            on:click={(e) => ($mode = Mode.MaskDraw)}><p>Magic Ink</p></button
        >
    </div>
{/if}
<div id="optionPanel">
    <SetImageOptions {mainCanvas} />
    <DrawMaskOptions {maskCanvas} />
    <DrawDirectOptions {mainCanvas} />
    <OptimizeOptions />
</div>

<style>
    #optionPanel {
        position: fixed;
        left: 0px;
        top: 69px;
        z-index: 2;
        width: 151px;
        display: flex;
        flex-direction: column;
        background: white;
        /* border-top-right-radius: 4px;
        border-bottom-right-radius: 4px; */
        border-right: 1px solid #ccc;
        border-bottom: 1px solid #ccc;
        align-items: center;
        overflow: hidden;
    }
    #modeToggle {
        position: fixed;
        top: 0px;
        left: 0px;
        background: white;
        z-index: 1;
        display: flex;
    }
    #modeToggle > button  {
        display: inline-block;
        padding: none;
        margin: 0px;
    }
    :global(button > img) {
        width: 40px;
        height: 40px;
        margin: 0px;

        user-drag: none;
        -webkit-user-drag: none;
        user-select: none;
        -moz-user-select: none;
        -webkit-user-select: none;
        -ms-user-select: none;
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
