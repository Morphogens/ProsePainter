<script lang="ts">
    import { mode, lastOptimizationResult } from "@/stores";
    import { Mode } from "../types";
    import * as optEvents from "../optimizeEvents";
    $: OR = $lastOptimizationResult;
    $: optimizeMessage = OR
        ? `${OR.step} / ${OR.num_iterations}`
        : "Waiting for server";
    $: optCompleted = OR && OR.step == OR.num_iterations
</script>

{#if $mode == Mode.Optimizing}
    <button on:click={() => optEvents.pause()}>
        <p>Stop</p>
    </button>
    <p>{optimizeMessage}</p>
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
{/if}

<style>
    button {
        width: 100%;
    }
</style>
