<script lang="ts">
    import InfiniteViewer from "../drawing/svelte-infinite-viewer";
    import DrawCanvas from "../drawing/DrawCanvas.svelte";
    import OptionPanel from "../drawing/OptionPanel.svelte";
    import LayersPanel from "../drawing/LayersPanel.svelte";
    import { activeLayer, activeLayerIdx, layers, addLayer } from "../drawing/stores";
    import { onMount } from "svelte";
    
    let previewCanvas: HTMLCanvasElement;
    let previewCanvasCtx: CanvasRenderingContext2D;

    const width = 512;
    const height = 512;

    addLayer('a tea pot')
    addLayer('a cat')

    activeLayerIdx.subscribe($activeLayerIdx => {
        if (!previewCanvasCtx) {
            return
        }
        for (const layer of $layers.slice().reverse()) {
            if (layer.data) {
                console.log('Drawing', layer.prompt);
                previewCanvasCtx.drawImage(layer.data, 0, 0);
            }
        }
    })

    onMount(() => {
        previewCanvasCtx = previewCanvas.getContext("2d");
    })
</script>

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
