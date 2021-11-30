<script lang="ts">
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import { mode } from "@/stores";
    import { Mode } from "../types";
    import Slider from "./Slider.svelte";
    export let mainCanvas: DrawCanvas;
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
{/if}
