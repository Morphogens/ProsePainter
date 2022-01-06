<script lang="ts">
    import { onMount } from "svelte";
    import { get } from "svelte/store";
    import InfiniteViewer from "@/drawing/svelte-infinite-viewer";
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import OptionPanel from "@/components/OptionPanel.svelte";
    import { mode } from "./stores";
    // import startBackgroundUrl from "./assets/startImage0.jpeg";
    import downloadUrl from "./assets/download.svg";
    import beforeAfterUrl from "./assets/before-after.svg";
    import { downloadCanvas, mergeCanvas } from "./utils";
    import { Mode } from "./types";
    import {
        lastOptimizationResult,
        mainCanvas,
        maskCanvas,
        canvasSize,
    } from "./stores";
    import { drawCircle, drawLines } from "./drawing/drawUtils";
    import { drawMaskGridAlpha, drawGrid } from "./maskDrawMethods";
    import InfoModal from "./components/InfoModal.svelte";
    import { DEFAULT_IMAGE } from "./constants";
    import { exportImage, drawHistoryStore } from "./drawHistory";

    let mouseover = false;
    let cursorCanvas: HTMLCanvasElement;
    let cursorCtx: CanvasRenderingContext2D;

    let outlineCanvas: HTMLCanvasElement;
    let outlineCtx: CanvasRenderingContext2D;
    $: magicMaskFilter = drawGrid($canvasSize, 3);
    let modal: InfoModal;

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
        if ($mode == Mode.SetImage) {
            return;
        }
        const canvas = activeDrawCanvas();
        cursorCtx.lineWidth = 2;
        cursorCtx.strokeStyle = "white";
        if (canvas) {
            cursorCtx.fillStyle = canvas.strokeColor;
            drawCircle(cursorCtx, x, y, canvas.radius, true, true);
        } else {
            cursorCtx.fillStyle = "black";
            drawCircle(cursorCtx, x, y, 2, true, true);
        }
    }

    function onMaskCanvasStroke(data) {
        const { ctx, canvas } = data.detail;
        drawMaskGridAlpha(
            outlineCtx,
            $canvasSize as [number, number],
            canvas,
            magicMaskFilter[0]
        );
    }

    function onMaskCanvasChange(data) {
        const { canvas, ctx } = data.detail;
        drawMaskGridAlpha(
            outlineCtx,
            $canvasSize as [number, number],
            canvas,
            magicMaskFilter[0]
        );
    }

    onMount(async () => {
        $mainCanvas.strokeColor = "#e66465";
        cursorCtx = cursorCanvas.getContext("2d");
        outlineCtx = outlineCanvas.getContext("2d");
    });

    // function localStorageSpace(){
    //     let data = '';

    //     console.log('Current local storage: ');

    //     for(let key in window.localStorage){
    //         if(window.localStorage.hasOwnProperty(key)){
    //             data += window.localStorage[key];
    //             console.log( key + " = " + ((window.localStorage[key].length * 16)/(8 * 1024)).toFixed(2) + ' KB' );
    //         }
    //     }
    //     console.log(data ? '\n' + 'Total space used: ' + ((data.length * 16)/(8 * 1024)).toFixed(2) + ' KB' : 'Empty (0 KB)');
    //     console.log(data ? 'Approx. space remaining: ' + (5120 - ((data.length * 16)/(8 * 1024)).toFixed(2)) + ' KB' : '5 MB');
    // };
    // localStorageSpace()
</script>

<svelte:window on:keydown={onKeyDown} />

<OptionPanel maskCanvas={$maskCanvas} mainCanvas={$mainCanvas} />

{#if $drawHistoryStore.hasOptimized}
    <button
        id="downloadBeforeAfterButton"
        class="roundCornerButton"
        on:click={async (e) => downloadCanvas(await exportImage(), 'before-after-prose.jpg')}
    >
        <img src={beforeAfterUrl} alt="download" />
    </button>
{/if}
<button
    id="downloadButton"
    class="roundCornerButton"
    on:click={(e) => downloadCanvas($mainCanvas.getCanvas(), 'prosepainter-image.png')}
>
    <img src={downloadUrl} alt="download" />
</button>

<InfoModal bind:this={modal} />
<button
    id="helpButton"
    class="roundCornerButton"
    on:click={(e) => modal.toggle()}
>
    ?
</button>

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
            <div style="opacity:1;">
                <DrawCanvas
                    {canvasSize}
                    radius={4}
                    id="mainCanvas"
                    defaultImageUrl={DEFAULT_IMAGE}
                    showCursor={$mode == Mode.SetImage}
                    bind:this={$mainCanvas}
                />
            </div>
            <div class:hidden={$mode == Mode.DirectDraw}>
                <div style="opacity:0;">
                    <DrawCanvas
                        radius={50}
                        softness={0.2}
                        {canvasSize}
                        id="maskCanvas"
                        maskFilter={magicMaskFilter[0]}
                        showCursor={$mode == Mode.SetImage}
                        on:change={onMaskCanvasChange}
                        on:stroke={onMaskCanvasStroke}
                        bind:this={$maskCanvas}
                    />
                </div>
            </div>
            <div class:hidden={$mode != Mode.MaskDraw}>
                <canvas
                    id="outlineCanvas"
                    class="hiddenOverlay"
                    width={$canvasSize[0]}
                    height={$canvasSize[1]}
                    bind:this={outlineCanvas}
                />
            </div>
            {#if $lastOptimizationResult}
                <img
                    id="optPreview"
                    class="hiddenOverlay"
                    src={$lastOptimizationResult.image.src}
                    alt=""
                />
            {/if}
            <canvas
                id="cursorCanvas"
                class="hiddenOverlay"
                width={$canvasSize[0]}
                height={$canvasSize[1]}
                bind:this={cursorCanvas}
            />
        </div>
    </div>
    <!-- {#if maskCanvasBase64}
        <img id='debugMask' src={maskCanvasBase64}>
    {/if} -->
</InfiniteViewer>

<style>
    :global(*) {
        font-family: "Open Sans", sans-serif;
        box-sizing: border-box;
    }
    :global(#maskCanvas, #optPreview, #cursorCanvas, #outlineCanvas) {
        top: 0px;
        left: 0px;
        position: absolute;
    }
    .hiddenOverlay {
        /* cursor: none; */
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
    :global(button) {
        border: 1px solid #ccc;
        display: inline-block;
        padding: 6px 12px;
        margin: 0px;
        /* background-color: white; */
        background-color: #f8f9fa;
        opacity: 1;

        cursor: pointer;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
    }
    :global(button:hover) {
        /* opacity: 0.8; */
        background-color: #f8f9fa79;
        /* border: 1px solid black; */
    }

    :global(button.selected) {
        color: white;
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
        margin-top: 100px;
        margin-left: 180px;
        background: white;
        box-shadow: -1px 4px 8px 0px rgba(0, 0, 0, 0.61);
    }
    .hidden {
        display: none;
    }
    .roundCornerButton {
        position: fixed;
        bottom: 16px;
        z-index: 1;
        background: white;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        padding: 8px;
    }
    .roundCornerButton img {
        width: 100%;
        height: 100%;
    }
    #helpButton {
        left: 12px;
    }
    #downloadButton {
        right: 16px;
        bottom: 16px;
    }
    #downloadBeforeAfterButton {
        width: 36px;
        height: 36px;
        right: 16px;
        bottom: 76px;
    }
</style>
