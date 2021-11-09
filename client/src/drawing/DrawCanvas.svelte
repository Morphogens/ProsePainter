<svelte:options accessors />

<script lang="ts">
    import { onMount, tick, createEventDispatcher } from "svelte";
    const dispatch = createEventDispatcher()
    import { drawLines } from "./drawUtils";
    // import { imageContour } from '../imageContour'
    import { loadImage } from "@/utils";
    // import { start } from "@/optimizeEvents";
    import { imgTob64 } from "@/utils";
    import { canvasSize }from '@/stores'

    export let id: string;
    export let radius = 20;
    export let opacity = 3;
    export let softness = 3;
    export let erasing = false;
    export let canvasBase64: string | null = null;
    export let strokeColor = "#000000";
    
    
    export let width: number;
    export let height: number;
    export let defaultImageUrl: string | null = null
    // export let 

    let canvas: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D;

    let currentStrokeCanvas: HTMLCanvasElement;
    let currentStrokeCtx: CanvasRenderingContext2D;

    // let outlinesCanvas: HTMLCanvasElement
    // let outlinesCtx: CanvasRenderingContext2D
    let undoStack: string[] = [];
    let undoStackMax = 30;
    let redoStack: string[] = [];
    let mouseDown = false;
    let mouseover = false;
    let currentStroke: number[][] = [];

    $: canvasBase64 = undoStack[undoStack.length - 1] ?? null
    
    function canvasChanged() {
        // Cleaning the oldest undoable object
        if (undoStack.length === undoStackMax) {
            undoStack.shift()
        }
        redoStack = []
        const dataURl = canvas.toDataURL()
        undoStack = [...undoStack, dataURl]
        // dispatch('canvasChanged', { ctx, canvas, canvasBase64 });
        window.localStorage.setItem(`${id}-canvasBase64`, dataURl)
    }

    export function clear() {
        ctx.clearRect(0, 0, width, height);
        canvasChanged();
    }

    export function set(src: HTMLCanvasElement | HTMLImageElement) {
        canvas.width = src.width
        canvas.height = src.height
        console.log('set image')
        ctx.drawImage(src, 0, 0)
        canvasChanged()
    }

    export function getCanvas(): HTMLCanvasElement {
        return canvas;
    }

    async function setFromUndoRedo() {
        ctx.clearRect(0, 0, width, height);
        if (undoStack.length) {
            const image = await loadImage(undoStack[undoStack.length-1]);
            ctx.drawImage(image, 0, 0);
        }
    }

    export async function undo() {
        const lastbase64 = undoStack.pop()
        await setFromUndoRedo()
        if (lastbase64) {            
            redoStack.push(lastbase64)
        }
    }
    export async function redo() {
        const lastbase64 = redoStack.pop()
        if (lastbase64) {
            undoStack.push(lastbase64);
        }
        await setFromUndoRedo()
        
    }
    onMount(async () => {
        ctx = canvas.getContext("2d");
        // Draw the start image that was saved
        currentStrokeCtx = currentStrokeCanvas.getContext("2d");
        // outlinesCtx = outlinesCanvas.getContext('2d')  
        const savedUrl = window.localStorage.getItem(`${id}-canvasBase64`)
        const startUrl = savedUrl || defaultImageUrl
        console.log('startUrl', startUrl.length);
         
        if (startUrl != null && startUrl != 'null') {
            const startImage = await loadImage(startUrl)
            ctx.drawImage(startImage, 0, 0)
            console.log(undoStack.length);
            
            // canvasChanged()
            // canvasBase64 = await imgTob64(startImage)
        }
    })

    // function drawContours(contours: number[][][]) {
    //     outlinesCtx.clearRect(0, 0, width, height)
    //     outlinesCtx.lineWidth = 1
    //     outlinesCtx.setLineDash([3, 3]);
    //     for (const poly of contours) {
    //         drawLines( outlinesCtx, poly)
    //     }
    // }
    // Listen for changes to the canvas and update contours.
    // $: if (canvasBase64.length && ctx) {
    //     drawContours(imageContour(canvas, ctx) as number[][][])
    // }

    function strokeDone() {
        if (!mouseDown) {
            return;
        }
        mouseDown = false;
        currentStroke = [];
        ctx.globalCompositeOperation = "source-over";
        ctx.drawImage(currentStrokeCanvas, 0, 0);
        currentStrokeCtx.clearRect(0, 0, width, height);
        canvasChanged();
    }

    function onMouseMove(event) {
        const [x, y] = [event.offsetX, event.offsetY];
        if (mouseDown) {
            currentStroke.push([x, y]);
            if (erasing) {
                ctx.filter = "none";
                ctx.globalCompositeOperation = "destination-out";
                ctx.strokeStyle = "rgba(255,255,255,1)";
                ctx.lineWidth = radius;
                ctx.lineCap = "round";
                ctx.lineJoin = "round";
                drawLines(ctx, currentStroke);
            } else {
                ctx.globalCompositeOperation = "destination-over";
                currentStrokeCtx.globalAlpha = 1 - opacity;
                currentStrokeCtx.filter = `blur(${softness}px)`;
                currentStrokeCtx.strokeStyle = strokeColor;
                currentStrokeCtx.lineWidth = radius;
                currentStrokeCtx.lineCap = "round";
                currentStrokeCtx.lineJoin = "round";
                currentStrokeCtx.clearRect(0, 0, width, height);
                drawLines(currentStrokeCtx, currentStroke);
            }
        }
    }
</script>

<div
    {id}
    class="canvasesContainer"
    on:mousedown={() => (mouseDown = true)}
    on:mouseup={() => strokeDone()}
    on:mouseleave={() => strokeDone()}
    on:mousemove={onMouseMove}
    on:mouseover={() => (mouseover = true)}
    on:mouseout={() => (mouseover = false)}
    on:focus={() => (mouseover = true)}
    on:blur={() => (mouseover = false)}
>
    <canvas id="drawCanvas" bind:this={canvas} {width} {height} />
    <!-- style='opacity:{mouseover ? 1 : 0}' -->
    <!-- <canvas
        id='outlinesCanvas'
        bind:this={outlinesCanvas}
        {width}
        {height}
        style='opacity:{mouseover ? 0 : 1}'
    /> -->
    <canvas
        id="currentStrokeCanvas"
        bind:this={currentStrokeCanvas}
        {width}
        {height}
    />
</div>

<style>
    .canvasesContainer {
        position: relative;
        cursor: none;
        /* cursor: crosshair; */
    }
    #currentStrokeCanvas {
        position: absolute;
        top: 0px;
        left: 0px;
    }
    /* .hidden */
</style>
