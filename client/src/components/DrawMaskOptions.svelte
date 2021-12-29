<script lang="ts">
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import {
        learningRate,
        prompt,
        mode,
        stylePrompt,
        numRecSteps,
        modelType,
    } from "@/stores";
    import * as optEvents from "../optimizeEvents";
    import { socketOpen } from "@/lib/socket";
    import { Mode } from "../types";
    import Slider from "./Slider.svelte";
    export let maskCanvas: DrawCanvas;
    let seeAdvanced = false
    import Select from 'svelte-select';

    const modelTypes = ['imagenet-16384', 'openimages-8192'];


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
        <img src="/images/pencil.svg" alt="draw-mask" />
    </button>
    <button
        on:click={() => (maskCanvas.erasing = true)}
        class:selected={maskCanvas.erasing}
    >
        <img src="/images/eraser.png" alt="erase-mask" />
    </button>
    <button on:click={() => maskCanvas.clear()}>
        <p>Clear</p>
    </button>
    <Slider
        name="Radius"
        bind:val={maskCanvas.radius}
        min={1}
        max={96}
        step={1}
    />
    <Slider
        name="Softness"
        bind:val={maskCanvas.softness}
        max={20}
        step={1}
    />
    {#if seeAdvanced}
        <Slider
            name="Learn Rate"
            bind:val={$learningRate}
            max={500}
            step={10}
        />
        <Slider
            name="Rec. Steps"
            bind:val={$numRecSteps}
            max={64}
            step={4}
        />
        <!-- Model Type
        <Select style="width:100%" items={modelTypes} bind:value={$modelType}></Select> -->

    {:else}
        <button on:click={() => seeAdvanced=true}>
            <p>Advanced</p>
        </button>
        <br>
    {/if}
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
    <br>
{/if}

<style>
    button {
        width: 100px;
    }

</style>