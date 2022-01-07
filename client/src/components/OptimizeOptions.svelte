<script lang="ts">
    import { mode, optimizationResults, selectedOptIdx, numQueueUsers } from "@/stores";
    import { Mode } from "../types";
    import * as optEvents from "../optimizeEvents";
    import Slider from './Slider.svelte'
    $: OR = $optimizationResults;
    $: optimizeMessage = OR
        ? `${OR.images.length} / ${OR.num_iterations}`
        : "Waiting for server";
    $: optCompleted = OR && OR.images.length == OR.num_iterations
</script>

{#if $mode == Mode.Optimizing}
    <button on:click={() => optEvents.pause()}>
        <p>Stop</p>
    </button>
    {#if OR}
        <p>{optimizeMessage}</p>
    {:else}
        {#if $numQueueUsers == 0}
            <p>You're up next!</p>
        {:else}
            <p>{$numQueueUsers} painters before you</p>
        {/if}
    {/if}
{:else if $mode == Mode.PausedOptimizing}
    <button on:click={() => optEvents.accept()}>
        <p>Accept</p>
    </button>
    <button on:click={() => optEvents.discard()}>
        <p>Discard</p>
    </button>
    <!-- <button on:click={() => optEvents.upscale()}>
        <p>Upscale</p>
    </button> -->
    {#if !optCompleted}
        <button on:click={() => optEvents.resume()}>
            <p>Resume</p>
        </button>
    {/if}
    <Slider name='step' min={0} max={$optimizationResults.images.length-1} step={1} bind:val={$selectedOptIdx}/>
{/if}

<style>
    button {
        width: 100%;
    }
</style>
