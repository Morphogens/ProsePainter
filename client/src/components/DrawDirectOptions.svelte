<script lang="ts">
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import { mode, canvasSize } from "@/stores";
    import { Mode } from "../types";
    import { loadImage } from "@/utils";
    import { tick } from "svelte";
    import Slider from "./Slider.svelte";
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
    <input type="color" bind:value={mainCanvas.strokeColor} />
    <Slider
        name="Radius"
        bind:val={mainCanvas.radius}
        min={1}
        max={96}
        step={1}
    />
    <Slider name="Softness" bind:val={mainCanvas.softness} max={20} step={1} />
    <p>Select an image</p>
    <input type="file" accept=".jpg, .jpeg, .png" on:change={onFiles} />
{/if}
