<script lang="ts">
    import type DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import { mode } from "@/stores";
    import { Mode } from "../types";
    import Slider from "./Slider.svelte";
    export let mainCanvas: DrawCanvas;

    const defaultColors = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        // "#008080",
        "#e6beff",
        "#9a6324",
        // "#fffac8",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000075",
        "#808080",
        "#ffffff",
        "#000000",
    ];
</script>

{#if $mode == Mode.DirectDraw && mainCanvas}
    
    <Slider
        name="Radius"
        bind:val={mainCanvas.radius}
        min={1}
        max={96}
        step={1}
    />
    <Slider name="Softness" bind:val={mainCanvas.softness} max={1} step={.05} />
    
    <br>
    <input type="color" bind:value={mainCanvas.strokeColor} />
    <div class='color-container'>
        {#each defaultColors as colorHex}
            <div class="color-opt" style="background:{colorHex};" on:click={() => mainCanvas.strokeColor = colorHex}/>
        {/each}
    </div>
{/if}

<style>
    input[type='color'] {
        cursor: pointer;
    }
    .color-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        width: 100%;
        /* padding: 2px; */
    }
    .color-opt {
        width: 30px;
        height: 30px;
        /* margin: 1px; */
        /* border: 1px solid; */
        cursor: pointer;
    }
</style>
