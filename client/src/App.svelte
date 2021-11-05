<script lang="ts">
    import { onMount } from "svelte";
    import { get } from "svelte/store";
    import InfiniteViewer from "@/drawing/svelte-infinite-viewer";
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import OptionPanel from "@/components/OptionPanel.svelte";
    import { mode } from "./stores";
    import startBackgroundUrl from "./assets/startImage0.jpeg";
    import downloadUrl from "./assets/download.svg";
    import { downloadCanvas, loadImage } from "./utils";
    import { Mode } from "./types";
    import {
        maskCanvasBase64,
        mainCanvasBase64,
        lastOptimizationResult,
        mainCanvas,
        maskCanvas,
        canvasSize,
    } from "./stores";


    function onKeyDown(e: KeyboardEvent) {
        const $mode = get(mode)
        if (e.code === "KeyZ" && (e.metaKey === true || e.ctrlKey === true)) {
            if (e.shiftKey) {
                if ($mode == Mode.DirectDraw) {
                    get(mainCanvas).redo()
                } else if ($mode == Mode.MaskDraw) {
                    get(maskCanvas).redo()
                }
            } else {
                if ($mode == Mode.DirectDraw) {
                    get(mainCanvas).undo()
                } else if ($mode == Mode.MaskDraw) {
                    get(maskCanvas).undo()
                }
            }
        }
    }

    function onKeyUp(e: KeyboardEvent) {}
    onMount(async () => {
        // $mainCanvas.set(await loadImage(startBackgroundUrl));
        $mainCanvas.strokeColor = "#e66465";
    });
    // log
</script>

<svelte:window on:keydown={onKeyDown} on:keyup={onKeyUp} />

<div id="modeToggle">
    <button
        class:selected={$mode == Mode.DirectDraw}
        on:click={(e) => ($mode = Mode.DirectDraw)}><p>Draw</p></button
    >
    <button
        class:selected={$mode == Mode.MaskDraw}
        on:click={(e) => ($mode = Mode.MaskDraw)}><p>Magic Draw</p></button
    >
</div>

<OptionPanel maskCanvas={$maskCanvas} mainCanvas={$mainCanvas} />

{#if $mode == Mode.DirectDraw || $mode == Mode.MaskDraw}
    <button
        id="downloadButton"
        on:click={(e) => downloadCanvas($mainCanvas.getCanvas())}
    >
        <img src={downloadUrl} alt="download" />
    </button>
{/if}

<InfiniteViewer
    className="viewer"
    usePinch={true}
    rangeX={[-256, 256]}
    rangeY={[-256, 256]}
>
    <div class="viewport" style="width:{$canvasSize[0]}px">
        <div
            id="content"
            style="width:{$canvasSize[0]}px;height:{$canvasSize[1]}px"
        >
            <DrawCanvas
                width={$canvasSize[0]}
                height={$canvasSize[1]}
                radius={4}
                id="mainCanvas"
                defaultImageUrl={startBackgroundUrl}
                bind:canvasBase64={$mainCanvasBase64}
                bind:this={$mainCanvas}
            />
            <div class:hidden={$mode == Mode.DirectDraw} style="opacity:0.5;">
                <DrawCanvas
                    width={$canvasSize[0]}
                    height={$canvasSize[1]}
                    id="maskCanvas"
                    bind:canvasBase64={$maskCanvasBase64}
                    bind:this={$maskCanvas}
                />
            </div>
            {#if $lastOptimizationResult}
                <img id="optPreview" src={$lastOptimizationResult.src} />
            {/if}
        </div>
    </div>
    <!-- {#if maskCanvasBase64}
        <img id='debugMask' src={maskCanvasBase64}>
    {/if} -->
</InfiniteViewer>

<style>
    :global(#maskCanvas, #optPreview) {
        top: 0px;
        left: 0px;
        position: absolute;
    }
    /* #debugMask  {
        position: fixed;
        bottom: 0px;
        right: 0px;
        border:1px solid black;
        width: 256px;
        height: 256px;
    } */

    .hidden {
        display: none;
    }
    #modeToggle {
        position: fixed;
        top: 0px;
        left: 0px;
        background: white;
        z-index: 1;
    }
    :global(button) {
        cursor: pointer;
        transition-duration: 0.2s;
        background-color: rgb(196, 196, 196);
        color: white;
        border: none;
        padding: 4px 16px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    :global(button.selected) {
        background-color: #4caf50; /* Green */
    }
    #content {
        display: flex;
    }
    :global(.viewer) {
        border: 1px solid black;
        top: 0px;
        left: 0px;
        position: absolute !important;
        width: 100vw;
        height: 100vh;
        background: gray;
    }
    .viewport {
        position: relative;
        margin: 100px;
        background: white;
        box-shadow: -1px 4px 8px 0px rgba(0, 0, 0, 0.61);
    }
    .hidden {
        display: none;
    }
    #downloadButton {
        position: fixed;
        right: 10px;
        bottom: 10px;
        z-index: 1;
        background: white;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        padding: 0px;
    }
</style>
