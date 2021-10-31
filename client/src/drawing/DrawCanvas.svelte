<script lang="ts">
    import { onMount, tick } from "svelte"
    import { drawLines, drawLine, drawCircle } from './drawUtils'
    import { erasing, radius, softness, opacity, saveState } from "./stores";
    import { imageContour } from '../imageContour'
    export let width: number
    export let height: number
    
    let canvas: HTMLCanvasElement
    let ctx: CanvasRenderingContext2D
    
    let currentStrokeCanvas: HTMLCanvasElement
    let currentStrokeCtx: CanvasRenderingContext2D
    
    let outlinesCanvas: HTMLCanvasElement
    let outlinesCtx: CanvasRenderingContext2D

    
    let contours = []
    let mouseDown = false
    let mouseover = false
    let drawing = false
    let currentStroke: number[][] = []

    $: if (drawing) {
        console.log("DRAWING!!")
    }
    
    onMount(() => {
        ctx = canvas.getContext('2d')
        outlinesCtx = outlinesCanvas.getContext('2d')
        currentStrokeCtx = currentStrokeCanvas.getContext('2d')
    })

    function drawContours(contours: number[][][]) {
        outlinesCtx.clearRect(0, 0, width, height)
        outlinesCtx.lineWidth = 1
        outlinesCtx.setLineDash([3, 3]);
        for (const poly of contours) {
            drawLines( outlinesCtx, poly)
        }
    }

    function strokeDone() {
        if (!mouseDown){
            return
        }
        mouseDown = false
        drawing = false
        currentStroke = []
        ctx.drawImage(currentStrokeCanvas, 0, 0)
        currentStrokeCtx.clearRect(0, 0, width, height)
        console.log("DRAW DONE")
        drawContours(imageContour(canvas, ctx) as number[][][])
        saveState(canvas)        
    }

    function onMouseMove(event) {
        const [x, y] = [event.offsetX, event.offsetY];
        if (mouseDown) {
            currentStroke.push([x, y])
            drawing = true
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
