<svelte:options accessors />

<script lang="ts">
    import { onMount, createEventDispatcher } from "svelte";
    const dispatch = createEventDispatcher();
    import { drawLines } from "./drawUtils";
    import { loadImage } from "../utils";

    export let id: string;
    export let radius = 30;
    export let opacity = 3;
    export let softness = 3;
    export let erasing = false;
    export let canvasBase64: string | null = null;
    export let strokeColor = "#000000";
    export let width: number;
    export let height: number;
    export let defaultImageUrl: string | null = null;
    export let maskFilter: HTMLCanvasElement | null = null; // Temporary hack.

    let canvas: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D;

    let currentStrokeCanvas: HTMLCanvasElement;
    let currentStrokeCtx: CanvasRenderingContext2D;

    let previewCanvas: HTMLCanvasElement = document.createElement("canvas");
    let previewCtx: CanvasRenderingContext2D = previewCanvas.getContext("2d");

    let undoStack: string[] = [];
    let undoStackMax = 30;
    let redoStack: string[] = [];
    let mouseDown = false;
    let currentStroke: number[][] = [];

    $: canvasBase64 = undoStack[undoStack.length - 1] ?? null;

    function canvasChanged() {
        if (undoStack.length === undoStackMax) {
            undoStack.shift();
        }
        redoStack = [];
        const dataURl = canvas.toDataURL();
        undoStack = [...undoStack, dataURl];
        dispatch("change", { ctx, canvas, canvasBase64 });
        window.localStorage.setItem(`${id}-canvasBase64`, dataURl);
    }

    export function clear() {
        ctx.clearRect(0, 0, width, height);
        canvasChanged();
    }

    export function resize(width: number, height: number) {
        canvas.width = width;
        canvas.height = height;
        currentStrokeCanvas.width = width;
        currentStrokeCanvas.height = height;
        previewCanvas.width = width;
        previewCanvas.height = height;
    }

    export function set(src: HTMLCanvasElement | HTMLImageElement) {
        resize(src.width, src.height);
        console.log("set image");
        ctx.drawImage(src, 0, 0);
        canvasChanged();
    }

    export function getCanvas(): HTMLCanvasElement {
        return canvas;
    }

    export function getContext(): CanvasRenderingContext2D {
        return ctx;
    }

    async function setFromUndoRedo() {
        ctx.clearRect(0, 0, width, height);
        if (undoStack.length) {
            const image = await loadImage(undoStack[undoStack.length - 1]);
            ctx.drawImage(image, 0, 0);
        }
        dispatch("change", { ctx, canvas, canvasBase64 });
    }

    export async function undo() {
        const lastbase64 = undoStack.pop();
        await setFromUndoRedo();
        if (lastbase64) {
            redoStack.push(lastbase64);
        }
    }
    export async function redo() {
        const lastbase64 = redoStack.pop();
        if (lastbase64) {
            undoStack.push(lastbase64);
        }
        await setFromUndoRedo();
    }
    onMount(async () => {
        ctx = canvas.getContext("2d");
        // Draw the start image that was saved
        currentStrokeCtx = currentStrokeCanvas.getContext("2d");
        const savedUrl = window.localStorage.getItem(`${id}-canvasBase64`);
        const startUrl = savedUrl || defaultImageUrl;
        console.log("startUrl", startUrl.length);

        resize(canvas.width, canvas.height);

        if (startUrl != null && startUrl != "null") {
            const startImage = await loadImage(startUrl);
            ctx.drawImage(startImage, 0, 0);
            console.log(undoStack.length);
            canvasChanged();
        }
    });

    function strokeDone() {
        if (!mouseDown) {
            return;
        }
        mouseDown = false;
        currentStroke = [];
        ctx.globalCompositeOperation = erasing
            ? "destination-out"
            : "source-over";
        ctx.drawImage(currentStrokeCanvas, 0, 0);
        currentStrokeCtx.clearRect(0, 0, width, height);
        canvasChanged();
        ctx.globalCompositeOperation = 'source-over'
    }

    function onMouseMove(event) {
        if (!mouseDown) {
            return;
        }
        const [x, y] = [event.offsetX, event.offsetY];
        currentStroke.push([x, y]);
        // currentStrokeCtx.globalAlpha = 1 - opacity;
        currentStrokeCtx.filter = `blur(${softness}px)`;
        currentStrokeCtx.strokeStyle = erasing
            ? "rgba(255,255,255,1)"
            : strokeColor;
        currentStrokeCtx.lineWidth = radius;
        currentStrokeCtx.lineCap = "round";
        currentStrokeCtx.lineJoin = "round";
        currentStrokeCtx.clearRect(0, 0, width, height);
        drawLines(currentStrokeCtx, currentStroke);
        previewCtx.clearRect(0, 0, canvas.width, canvas.height);
        previewCtx.drawImage(canvas, 0, 0);
        previewCtx.globalCompositeOperation = erasing
            ? "destination-out"
            : "destination-over";
        previewCtx.drawImage(currentStrokeCanvas, 0, 0);
        dispatch("stroke", {
            ctx: previewCtx,
            canvas: previewCanvas,
        });
        previewCtx.globalCompositeOperation = "source-over"; // reset.
    }
</script>

<div
    {id}
    class="canvasesContainer"
    on:mousedown={() => (mouseDown = true)}
    on:mouseup={() => strokeDone()}
    on:mouseleave={() => strokeDone()}
    on:mousemove={onMouseMove}
>
    <canvas id="drawCanvas" bind:this={canvas} {width} {height} />
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
    }
    #currentStrokeCanvas {
        position: absolute;
        top: 0px;
        left: 0px;
    }
</style>
