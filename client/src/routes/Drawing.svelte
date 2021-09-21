<script lang="ts">
    import InfiniteViewer from "../drawing/svelte-infinite-viewer";
    import DrawCanvas from "../drawing/DrawCanvas.svelte";
    import OptionPanel from "../drawing/OptionPanel.svelte";
    import LayersPanel from "../drawing/LayersPanel.svelte";
    import { activeLayer, activeLayerIdx, layers, addLayer, undo, layerImages } from "../drawing/stores";
    import { onMount, tick } from "svelte";
    
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
    <div class="viewport" style="width:{width}px;height:{height}px">
        <canvas
            id='previewCanvas'
            bind:this={previewCanvas}
            {width}
            {height}
            style="opacity:{$activeLayer ? .33 : 1.0};"
        />
        {#if $activeLayer}
            <DrawCanvas {width} {height} />
        {/if}
    </div>
</InfiniteViewer>

<style>
    #previewCanvas {
        top:0px;
        position:absolute;
        pointer-events:none;
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
