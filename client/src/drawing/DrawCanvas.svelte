<script lang="ts">
    import { onMount } from "svelte"
    import { drawLines } from './drawUtils'
    import { erasing, radius, softness, opacity, canvasChanged, drawContext, canvasBase64 } from "./stores";
    import { imageContour } from '../imageContour'
    import { loadImage } from "@/utils";
    export let width: number
    export let height: number
    
    let canvas: HTMLCanvasElement
    let ctx: CanvasRenderingContext2D
    
    let currentStrokeCanvas: HTMLCanvasElement
    let currentStrokeCtx: CanvasRenderingContext2D
    
    let outlinesCanvas: HTMLCanvasElement
    let outlinesCtx: CanvasRenderingContext2D

    let mouseDown = false
    let mouseover = false
    let currentStroke: number[][] = []

    onMount(async () => {
        ctx = canvas.getContext('2d')
        outlinesCtx = outlinesCanvas.getContext('2d')
        currentStrokeCtx = currentStrokeCanvas.getContext('2d')
        drawContext.set({ canvas, ctx })
        // Draw the start image that was saved
        const last = window.localStorage.getItem('canvasBase64')
        if (last) {
            ctx.drawImage(await loadImage(last), 0, 0)
            canvasBase64.set(last)
            drawContours(imageContour(canvas, ctx) as number[][][])
        }
    })
    
    function drawContours(contours: number[][][]) {
        outlinesCtx.clearRect(0, 0, width, height)
        outlinesCtx.lineWidth = 1
        outlinesCtx.setLineDash([3, 3]);
        for (const poly of contours) {
            drawLines( outlinesCtx, poly)
        }
    }

    // Listen for changes to the canvas and update contours.
    canvasBase64.subscribe(($canvasBase64) => {
        if ($canvasBase64) {
            drawContours(imageContour(canvas, ctx) as number[][][])
        }
    })
    
    function strokeDone() {
        if (!mouseDown){
            return
        }
        mouseDown = false
        currentStroke = []
        ctx.drawImage(currentStrokeCanvas, 0, 0)
        currentStrokeCtx.clearRect(0, 0, width, height)
        canvasChanged()
    }

    function onMouseMove(event) {
        const [x, y] = [event.offsetX, event.offsetY];
        if (mouseDown) {
            currentStroke.push([x, y])
            if ($erasing) {
                ctx.filter = 'none';
                ctx.globalCompositeOperation = 'destination-out'
                ctx.strokeStyle = 'rgba(255,255,255,1)'
                ctx.lineWidth = $radius
                ctx.lineCap = 'round'
                ctx.lineJoin = 'round'
                drawLines(ctx, currentStroke)
            } else {
                ctx.globalCompositeOperation = 'destination-over'
                currentStrokeCtx.globalAlpha = 1 - $opacity;
                currentStrokeCtx.filter = `blur(${$softness}px)`;
                currentStrokeCtx.strokeStyle = 'black'
                currentStrokeCtx.lineWidth = $radius
                currentStrokeCtx.lineCap = 'round'
                currentStrokeCtx.lineJoin = 'round'
                currentStrokeCtx.clearRect(0, 0, width, height)
                drawLines(currentStrokeCtx, currentStroke)
            }            
        }
    }

</script>
<div id='canvasesContainer'
    on:mousedown={() => (mouseDown = true)}
    on:mouseup={() => strokeDone()}
    on:mouseleave={() => strokeDone()}
    on:mousemove={onMouseMove}
    on:mouseover={() => mouseover=true}
    on:mouseout={() => mouseover=false}
    on:focus={() => mouseover=true}
    on:blur={() => mouseover=false}
>
    <canvas
        id='drawCanvas'
        bind:this={canvas}
        {width}
        {height}
        style='opacity:{mouseover ? 1 : 0}'
    />
    <canvas
        id='outlinesCanvas'
        bind:this={outlinesCanvas}
        {width}
        {height}
        style='opacity:{mouseover ? 0 : 1}'
    />
    <canvas
        id="currentStrokeCanvas"
        bind:this={currentStrokeCanvas}
        {width}
        {height}
    />
</div>


<style>
    #canvasesContainer {
        position: relative;
        cursor: crosshair;
    }
    #currentStrokeCanvas, #outlinesCanvas {
        position: absolute;
        top: 0px;
        left: 0px;
    }
    /* .hidden */
</style>
