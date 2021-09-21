<script lang="ts">
    import { onMount } from "svelte";
    import { activeLayer, erasing, radius, layerImages } from "./stores";
    import { drawLine, drawCircle } from './drawUtils'
    export let width: number
    export let height: number
    let canvas: HTMLCanvasElement
    let ctx: CanvasRenderingContext2D
    let mouseDown = false
    let lastMouse: null | [number, number] = null 

    function onMouseMove(event) {
        const [x, y] = [event.offsetX, event.offsetY];
        if (mouseDown) {
            if ($erasing) {
                ctx.globalCompositeOperation = "destination-out";
                ctx.strokeStyle = "rgba(255,255,255,1)";
            } else {
                ctx.globalCompositeOperation = "source-over";
                ctx.strokeStyle = $activeLayer.get('color');
                ctx.fillStyle = $activeLayer.get('color');
            }
            drawCircle(ctx, x, y, $radius)
            if (lastMouse) {
                drawLine(ctx, ...lastMouse, x, y, $radius*2)
            }
            lastMouse = [x, y]
        }
    }

    function drawDone() {
        mouseDown = false
        lastMouse = null
        const imageBase64 = canvas.toDataURL('image/png')
        const image = new Image(width, height)
        image.src = imageBase64

        $activeLayer.set('imageBase64', imageBase64)
    }
    
    
    let lastActiveLayer = null
    $: if ($activeLayer != lastActiveLayer && ctx) {
        ctx.clearRect(0, 0, width, height)
        lastActiveLayer = $activeLayer        
        if ($activeLayer) {
            const image = layerImages.get($activeLayer)
            if (image) {
                ctx.drawImage(image, 0, 0)
            }
        }
    }

    onMount(() => {
        ctx = canvas.getContext("2d");
    });
</script>

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
