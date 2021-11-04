<script lang="ts">
    (window as any).global = window;
    import InfiniteViewer from "@/drawing/svelte-infinite-viewer";
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import OptionPanel from "@/components/OptionPanel.svelte";
    // import { socket, socketOpen } from "@/lib/socket";
    import { onMount, tick } from "svelte";
    // import Indicator from "@/components/Indicator.svelte";
    import { mode } from "./stores";
    import startBackgroundUrl from "./assets/startImage0.jpeg";
    import { loadImage } from "./utils";
    import { Mode } from "./types";
    import { maskCanvasBase64, mainCanvasBase64, lastOptimizationResult, mainCanvas, maskCanvas } from "./stores";
    const width = 512;
    const height = 512;
    // let mainCanvas: DrawCanvas;
    // let maskCanvas: DrawCanvas;
    // let setMainCanvas: Function
    
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
    onMount(async () => {
        $mainCanvas.set(await loadImage(startBackgroundUrl));
        $mainCanvas.strokeColor = "#e66465" 
    });
</script>

<svelte:window on:keydown={onKeyDown} on:keyup={onKeyUp} />

<!-- <div class="flex items-center px-2 py-1">
    <Indicator state={$socketOpen} />
    <div class="ml-1">Connection is {$socketOpen ? "open" : "closed"}</div>
</div> -->

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

<InfiniteViewer
    className="viewer"
    usePinch={true}
    rangeX={[-256, 256]}
    rangeY={[-256, 256]}
>
    <div class="viewport" style="width:{width}px">
        <div id="content" style="width:{width}px;height:{height}px">
            <DrawCanvas
                {width}
                {height}
                radius={4}
                id="mainCanvas"
                bind:canvasBase64={$mainCanvasBase64}
                bind:this={$mainCanvas}
            />
            <!-- bind:set={setMainCanvas} -->
            <div class:hidden={$mode == Mode.DirectDraw} style='opacity:0.5;'>
                <DrawCanvas
                    {width}
                    {height}
                    id="maskCanvas"
                    bind:canvasBase64={$maskCanvasBase64}
                    bind:this={$maskCanvas}
                />
            </div>
            {#if $lastOptimizationResult}
                <img id='optPreview' src={$lastOptimizationResult.src}/>
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
    #debugMask  {
        position: fixed;
        bottom: 0px;
        right: 0px;
        border:1px solid black;
        width: 256px;
        height: 256px;
    }
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
        background-color: #4CAF50; /* Green */
    }

    /* #backgroundCanvas { */
    /* } */
    /* #previewCanvas {
        top: 0px;
        position: absolute;
        pointer-events: none;
    } */
    #content {
        display: flex;
        /* border-bottom: 1px dashed;    */
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
