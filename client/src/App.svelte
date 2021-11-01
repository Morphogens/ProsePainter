<script lang="ts">
    (window as any).global = window;
    import InfiniteViewer from "@/drawing/svelte-infinite-viewer";
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import OptionPanel from "@/components/OptionPanel.svelte";
    import { socket, socketOpen } from "@/lib/socket";
    import { onMount, tick } from "svelte";
    import Indicator from "@/components/Indicator.svelte";
    import { lastOptimizationResult, isOptimizing, startGeneration, stopGeneration } from './stores'
    import { canvasBase64 } from "./drawing/stores";

    let backgroundCanvas:HTMLCanvasElement
    let backgroundCanvasCTX:CanvasRenderingContext2D
    const width = 512
    const height = 512
    
    $: if (backgroundCanvas && $lastOptimizationResult) {
        backgroundCanvasCTX.drawImage($lastOptimizationResult, 0, 0)
    }

    function onKeyDown(e: KeyboardEvent) {
        // if (e.code === "KeyZ" && (e.metaKey === true || e.ctrlKey === true)) {
        //     if (e.shiftKey) {
        //         undo.redo();
        //     } else {
        //         undo.undo();
        //     }
        // }
    }

    function onKeyUp(e: KeyboardEvent) {}
    onMount(() => {
        backgroundCanvasCTX = backgroundCanvas.getContext('2d')
    })
</script>

<svelte:window on:keydown={onKeyDown} on:keyup={onKeyUp} />
<div class="flex items-center px-2 py-1">
    <Indicator state={$socketOpen} />
    <div class="ml-1">Connection is {$socketOpen ? "open" : "closed"}</div>
</div>
<OptionPanel />

<InfiniteViewer
    className="viewer"
    usePinch={true}
    rangeX={[-256, 256]}
    rangeY={[-256, 256]}
>
    <div class="viewport" style="width:{width}px">
        <div id="content" style="width:{width}px;height:{height}px">
            <!-- <img bind:this={backgroundImage} alt=''> -->
            <canvas id ='backgroundCanvas' bind:this={backgroundCanvas} {width} {height} />
            <DrawCanvas {width} {height}/>
        </div>
        <div id="canvasButtons">
            {#if $isOptimizing}
                <button on:click={() => (stopGeneration())}> Stop </button>
            {:else if $socketOpen}
                <button on:click={() => (startGeneration())}> Start </button>
            {:else}
                <button on:click={() => (alert("SOCKET NOT CONNECTED :("))}> Start </button>
                <!-- <p> Socket not open </p> -->
            {/if}
        </div>
    </div>
    <img id='debugMask' src={$canvasBase64}>
</InfiniteViewer>

<style>
    #debugMask  {
        position: fixed;
        bottom: 0px;
        right: 0px;
        border:1px solid black;
        width: 256px;
    }
    #backgroundCanvas {
        top: 0px;
        position: absolute;
    }
    /* #previewCanvas {
        top: 0px;
        position: absolute;
        pointer-events: none;
    } */
    #content {
        display: flex;
        border-bottom: 1px dashed;
        
    }
    #canvasButtons {
        display: flex;
        justify-content: space-between;
    }
    :global(.viewer) {
        border: 1px solid black;
        /* position: relative; */
        top: 0px;
        left: 0px;
        position: absolute !important;
        width: 100vw;
        height: 100vh;
        background: gray;
    }
    :global(.viewer) {
        
    }
    .viewport {
        
        /* pointer-events: none; */
        position: relative;
        margin: 100px;
        background: white;
        box-shadow: -1px 4px 8px 0px rgba(0, 0, 0, 0.61);
    }
    .hidden {
        display: none;
    }
</style>
