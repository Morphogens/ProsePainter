<script lang="ts">
    import { onMount, tick } from "svelte";
    import { activeLayer, layers, erasing, radius, layerImages } from "./stores";
    import { drawLine, drawCircle } from './drawUtils'

    export let width: number
    export let height: number
    export let backgroundImage: HTMLImageElement
    export let run = false;
    export let showLayers = true;

    let canvas: HTMLCanvasElement
    let ctx: CanvasRenderingContext2D
    let canvasReady: boolean = false
    
    let layerCanvas: HTMLCanvasElement
    let layerCtx: CanvasRenderingContext2D
    let layerCanvasReady: boolean = false

    let mouseDown = false
    let lastMouse: null | [number, number] = null 
    
    let lastActiveLayer = null

    $: if(drawing){
        console.log("DRAWING!!")
        console.log(drawing)
    }
    
    onMount(() => {
        ctx = canvas.getContext("2d");
        canvasReady = true

        layerCtx = layerCanvas.getContext("2d");
        layerCanvasReady = true
    });


    let drawing = false
    
    async function drawBackground() {
        await tick();

        if (!ctx) {
            return;
        }

        if (backgroundImage.src == ""){
            console.log("RESETTING BACKGROUND")
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, width, height);
            backgroundImage.src = canvas.toDataURL('image/png')

        } else {
            console.log("DRAWING BACKGROUND")
            ctx.drawImage(backgroundImage, 0, 0, width, height)
        }
    }
    
    async function drawLayers(){
        console.log("DRAWING LAYERS")
        await tick();

        if (!ctx) {
            return;
        }

        for (const layer of $layers.slice()) {
            const image = layerImages.get(layer);
            console.log("DRAWING LAYER")
            ctx.drawImage(image, 0, 0);
            // console.log(layer == $activeLayer, layer, $activeLayer);
            // if (image && layer != $activeLayer) {
            //     console.log("DRAWING LAYER")
            //     ctx.drawImage(image, 0, 0);
            // }
        }
    }


    function drawDone() {
        if (!mouseDown){
            return
        }
        
        if (!activeLayer){
            return
        }

        console.log("DRAW DONE")

        lastMouse = null

        const imageBase64 = layerCanvas.toDataURL('image/png')
        const image = new Image(width, height)
        image.src = imageBase64

        console.log("ACTIVE IMAGE RESETED!!")
        $activeLayer.set('imageBase64', imageBase64)

        mouseDown = false
        drawing = false

        drawCanvas()
    }

    function onMouseMove(event) {
        if (!$activeLayer){
            return
        }

        const [x, y] = [event.offsetX, event.offsetY];
        if (mouseDown) {
            drawing = true

            if ($erasing) {
                layerCtx.globalCompositeOperation = "destination-out";
                layerCtx.strokeStyle = "rgba(255,255,255,1)";
            } else {
                layerCtx.globalCompositeOperation = "source-over";
                layerCtx.strokeStyle = $activeLayer.get('color');
                layerCtx.fillStyle = $activeLayer.get('color');
            }
            drawCircle(layerCtx, x, y, $radius)
            if (lastMouse) {
                drawLine(layerCtx, ...lastMouse, x, y, $radius*2)
            }
            lastMouse = [x, y]
        }
    }
    
    $: if ($activeLayer != lastActiveLayer && layerCtx) {
        console.log("ACTIVE LAYER CHANGED!")
        // layerCtx.clearRect(0, 0, width, height)
        lastActiveLayer = $activeLayer        
        if ($activeLayer) {
            const image = layerImages.get($activeLayer)
            if (image) {
                layerCtx.drawImage(image, 0, 0)
            }
        }
    }

    async function drawCanvas(){
        await drawBackground()
        if (showLayers){
            await drawLayers()
        }
    }

    $: if(true){
        console.log(showLayers)
        drawCanvas()
    }
    
    $: if ($layers || $activeLayer ) {
        if ($activeLayer){
            drawCanvas()
        }
    }
    
    // $: if (backgroundImage && ctx) {
    //     // ctx.clearRect(0, 0, width, height)
    //     console.log("BACKGROUND CHANGED!")
    //     // drawBackground()
    // }

</script>

<!-- {#if drawing || !layerCanvasReady}
    drawing canvas
    <canvas
        bind:this={layerCanvas}
        {width}
        {height}
        on:mousedown={() => (mouseDown = true)}
        on:mouseup={() => drawDone()}
        on:mouseleave={() => drawDone()}
        on:mousemove={onMouseMove}
    />
{/if}

{#if !drawing || !canvasReady}
    regular canvas
    <canvas
        bind:this={canvas}
        {width}
        {height}
        on:mousedown={() => (mouseDown = true)}
        on:mouseup={() => drawDone()}
        on:mouseleave={() => drawDone()}
        on:mousemove={onMouseMove}
    />
{/if} -->
    
drawing canvas
<canvas
    bind:this={layerCanvas}
    {width}
    {height}
    on:mousedown={() => (mouseDown = true)}
    on:mouseup={() => drawDone()}
    on:mouseleave={() => drawDone()}
    on:mousemove={onMouseMove}
/>

regular canvas
<canvas
    bind:this={canvas}
    {width}
    {height}
    on:mousedown={() => (mouseDown = true)}
    on:mouseup={() => drawDone()}
    on:mouseleave={() => drawDone()}
    on:mousemove={onMouseMove}
/>

<style>
    canvas {
        position: relative;
        cursor: crosshair;
    }
</style>
