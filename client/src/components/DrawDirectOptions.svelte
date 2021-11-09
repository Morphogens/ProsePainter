<script lang="ts">
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import { mode, canvasSize } from "@/stores";
    import { Mode } from "../types";
    import { loadImage } from "@/utils";
    import { tick } from "svelte";
    // export let maskCanvas: DrawCanvas;
    export let mainCanvas: DrawCanvas;

    async function onFiles(event) {
        const { files } = event.target;
        if (files.length) {
            const url = URL.createObjectURL(files[0]);
            const image = await loadImage(url);
            canvasSize.set([image.width, image.height]);
            await tick(); // DrawCanvases get recreated.
            mainCanvas.set(image);
        }
    }
</script>

{#if $mode == Mode.DirectDraw && mainCanvas}
    <input
        type="color"
        id="head"
        name="head"
        bind:value={mainCanvas.strokeColor}
    />
    <label for="head">Head</label>
    <p>Radius</p>
    <input type="range" bind:value={mainCanvas.radius} min="1" max="96" />
    {mainCanvas.radius}
    <p>Softness</p>
    <input
        type="range"
        bind:value={mainCanvas.softness}
        min="0"
        max="20"
        step=1
    />
    {mainCanvas.softness}
    <p>Select an image</p>
    <input type="file" accept=".jpg, .jpeg, .png" on:change={onFiles} />
{/if}
