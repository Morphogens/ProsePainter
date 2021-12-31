<script lang="ts">
    import type DrawCanvas from "@/drawing/DrawCanvas.svelte";
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
    <!-- <input type="text" minlength="3" bind:value={$prompt} /> -->
    <textarea rows="3" class="auto_height" bind:value={$prompt} />

    <!-- <p>In what style?</p>
    <input type="text" placeholder="Default" bind:value={$stylePrompt} /> -->
    <div class="button-group">
        <button
            on:click={() => (maskCanvas.erasing = false)}
            class:selected={!maskCanvas.erasing}
        >
            <!-- style="width:80%;" -->
            <img src="/images/brush.svg" alt="draw-mask" />
        </button>
        <button
            on:click={() => (maskCanvas.erasing = true)}
            class:selected={maskCanvas.erasing}
        >
            <img src="/images/eraser.png" alt="erase-mask" />
        </button>
    </div>
    <Slider
        name="Radius"
        bind:val={maskCanvas.radius}
        min={1}
        max={96}
        step={1}
    />
    <Slider name="Softness" bind:val={maskCanvas.softness} max={20} step={1} />
    <button on:click={() => maskCanvas.clear()}> Clear </button>
    
    {#if seeAdvanced}
        <Slider
            name="Learn Rate"
            bind:val={$learningRate}
            min={10}
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

        <p class="advanced" on:click={() => (seeAdvanced = false)}>
            Hide Advanced
        </p>

    {:else}
        <p class="advanced" on:click={() => (seeAdvanced = true)}>Advanced</p>
    {/if}
    {#if $socketOpen}
        <button
            on:click={() => optEvents.start()}
            style="width:100%;background-color: #4caf50;border:none;font-size:20px;"
        >
        Start
            <!-- <h4 style=margin:12px;> -->

            <!-- </h4> -->
        </button>
    {:else}
        <button on:click={() => alert("SOCKET NOT CONNECTED :(")}>
            Start
        </button>
        <p>Socket not open</p>
    {/if}
{/if}

<style>
    .button-group {
        display: flex;
        border-radius: 30px;
        overflow: hidden;
        border: 1px solid #ccc;
    }
    .button-group button {
        border: 0px;
    }
    .advanced {
        cursor: pointer;
        text-decoration: underline;
        margin: 16px !important;
    }
    textarea {
        resize: none;
        width: 90%;
        margin: 4px;
    }
</style>
