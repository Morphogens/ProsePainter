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
        lastOptimizationResult,
        mainCanvas,
        maskCanvas,
        canvasSize,
    } from "./stores";
    import { drawCircle, drawLines } from "./drawing/drawUtils";
    import { imageContour } from "./imageContour";

    let mouseover = false;
    let cursorCanvas: HTMLCanvasElement;
    let cursorCtx: CanvasRenderingContext2D;

    let outlineCanvas: HTMLCanvasElement;
    let outlineCtx: CanvasRenderingContext2D;

    function activeDrawCanvas(): DrawCanvas {
        if ($mode == Mode.DirectDraw) {
            return get(mainCanvas);
        } else if ($mode == Mode.MaskDraw) {
            return get(maskCanvas);
        }
    }

    function onKeyDown(e: KeyboardEvent) {
        if (e.code === "KeyZ" && (e.metaKey === true || e.ctrlKey === true)) {
            const activeCanvas = activeDrawCanvas();
            if (e.shiftKey) {
                activeCanvas.redo();
            } else {
                activeCanvas.undo();
            }
        }
    }

    $: if (!mouseover && cursorCtx) {
        cursorCtx.clearRect(0, 0, $canvasSize[0], $canvasSize[1]);
    }

    function onMouseMove(event) {
        const [x, y] = [event.offsetX, event.offsetY];
        cursorCtx.clearRect(0, 0, $canvasSize[0], $canvasSize[1]);
        const canvas = activeDrawCanvas();
        cursorCtx.lineWidth = 1;
        cursorCtx.strokeStyle = "white";
        if (canvas) {
            cursorCtx.fillStyle = canvas.strokeColor;
            drawCircle(
                cursorCtx,
                x,
                y,
                (canvas.radius + canvas.softness) / 2,
                true,
                true
            );
        } else {
            cursorCtx.fillStyle = "black";
            drawCircle(cursorCtx, x, y, 2, true, true);
        }
    }

    function drawContours(contours: number[][][]) {
        outlineCtx.clearRect(0, 0, $canvasSize[0], $canvasSize[1]);
        outlineCtx.lineWidth = 1;
        outlineCtx.setLineDash([3, 3]);
        for (const poly of contours) {
            drawLines(outlineCtx, poly);
        }
    }

    function onMaskCanvasChange(data) {
        const { canvas, ctx } = data.detail;
        drawContours(imageContour(canvas, ctx) as number[][][]);
    }

    onMount(async () => {
        $mainCanvas.strokeColor = "#e66465";
        cursorCtx = cursorCanvas.getContext("2d");
        outlineCtx = outlineCanvas.getContext("2d");
    });
</script>

<svelte:window on:keydown={onKeyDown} />

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
            on:mousemove={onMouseMove}
            on:mouseover={() => (mouseover = true)}
            on:mouseout={() => (mouseover = false)}
            on:focus={() => (mouseover = true)}
            on:blur={() => (mouseover = false)}
        >
            <DrawCanvas
                width={$canvasSize[0]}
                height={$canvasSize[1]}
                radius={4}
                id="mainCanvas"
                defaultImageUrl={startBackgroundUrl}
                bind:this={$mainCanvas}
            />
            <div
                class:hidden={$mode == Mode.DirectDraw || !mouseover}
                style="opacity:0.5;"
            >
                <DrawCanvas
                    width={$canvasSize[0]}
                    height={$canvasSize[1]}
                    id="maskCanvas"
                    on:change={onMaskCanvasChange}
                    bind:this={$maskCanvas}
                />
            </div>
            <canvas
                id="outlineCanvas"
                width={$canvasSize[0]}
                height={$canvasSize[1]}
                bind:this={outlineCanvas}
            />
            {#if $lastOptimizationResult}
                <img id="optPreview" src={$lastOptimizationResult.src} />
            {/if}
            <canvas
                id="cursorCanvas"
                width={$canvasSize[0]}
                height={$canvasSize[1]}
                bind:this={cursorCanvas}
            />
            <!-- style="width:{$canvasSize[0]}px;height:{$canvasSize[1]}px;pointer-events:none;" -->
        </div>
    </div>
    <!-- {#if maskCanvasBase64}
        <img id='debugMask' src={maskCanvasBase64}>
    {/if} -->
</InfiniteViewer>

<style>
    :global(#maskCanvas, #optPreview, #cursorCanvas, #outlineCanvas) {
        top: 0px;
        left: 0px;
        position: absolute;
    }
    #cursorCanvas,
    #outlineCanvas {
        cursor: none;
        pointer-events: none;
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
