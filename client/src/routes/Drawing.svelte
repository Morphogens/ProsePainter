<script lang="ts">
    import InfiniteViewer from "@/drawing/svelte-infinite-viewer";
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import OptionPanel from "@/drawing/OptionPanel.svelte";
    import LayersPanel from "@/drawing/LayersPanel.svelte";
    import { socket, socketOpen } from '@/lib/socket';
    import { activeLayer, layers, undo, layerImages } from "@/drawing/stores";
    import { onMount, tick } from "svelte";
    import Indicator from "@/Indicator.svelte";
    
    let isOptimizing = false
    let previewCanvas: HTMLCanvasElement;
    let previewCanvasCtx: CanvasRenderingContext2D;

    const width = 512;
    const height = 512;

    async function drawBackground() {
        await tick()
        if (!previewCanvasCtx) {
            return
        }
        previewCanvasCtx.clearRect(0, 0, width, height)
        for (const layer of $layers.slice().reverse()){
            const image = layerImages.get(layer)
            // console.log(layer == $activeLayer, layer, $activeLayer);
            if (image && layer != $activeLayer) {
                previewCanvasCtx.drawImage(image, 0, 0)
            }
        }
    }

    $: if ($layers || $activeLayer) {
        drawBackground()
    }
    
    function sendMessage(topic: string, data: any) {
        if ($socketOpen) {
            socket.send(JSON.stringify({ topic, data }));
        }
    }

    $: if (isOptimizing) {
        const data = $layers.map(l => l.toJSON())        
        sendMessage('setState', data)
    } else {
        sendMessage('setState', 'paused')
    }

    let img : string | undefined = undefined
    socket.addEventListener('message', (e) => {
        const message = JSON.parse(e.data);
        if (message.image) {
            img = 'data:text/plain;base64,' + message.image
        }
    });
    
    function onKeyDown(e: KeyboardEvent) {
        if (e.code === 'KeyZ' && (e.metaKey === true || e.ctrlKey === true)) {
            if (e.shiftKey) {
                undo.redo()
            } else {
                undo.undo()
            }
        }
    }
    function onKeyUp(e: KeyboardEvent) {}
    onMount(() => {
        previewCanvasCtx = previewCanvas.getContext("2d");
    })    
</script>

<svelte:window  on:keydown={onKeyDown} on:keyup={onKeyUp} />
<div class="flex items-center px-2 py-1">
    <Indicator state={$socketOpen} />
    <div class="ml-1">Connection is {$socketOpen ? 'open' : 'closed'}</div>
  </div>
{#if $activeLayer}
    <OptionPanel />
{/if}
<LayersPanel />
<InfiniteViewer
    className="viewer"
    usePinch={true}
    rangeX={[-256, 256]}
    rangeY={[-256, 256]}
>
    <div class="viewport" style="width:{width}px">
        <div id='content' style="width:{width}px;height:{height}px">
            <img class="select-none" draggable="false" src={img} />
            <canvas
                id='previewCanvas'
                bind:this={previewCanvas}
                {width}
                {height}
                style="opacity:{$activeLayer ? .33 : .33};"
            />
            {#if $activeLayer}
                <DrawCanvas {width} {height} />
            {/if}
        </div>
        {#if isOptimizing}
            <button on:click={() => isOptimizing = false}> Stop </button>
        {:else}
            {#if $socketOpen}
                <button on:click={() => isOptimizing = true}> Start </button>
            {:else}
                <button on:click={() => isOptimizing = true}> Start </button>
                <!-- <p> Socket not open </p> -->
            {/if}
        {/if}
    </div>
</InfiniteViewer>

<style>
    #previewCanvas {
        top:0px;
        position:absolute;
        pointer-events:none;
    }
    #content {
        border-bottom: 1px dashed;
    }
    :global(.viewer) {
        border: 1px solid black;
        position: relative;
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
</style>
