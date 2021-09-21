<script lang="ts">
    import { layers, activeLayer, activeLayerIdx, addLayer } from "./stores";
    
    function onClick() {
        const userPrompt = prompt("Enter a prompt");
        if (userPrompt) {
            addLayer(userPrompt)
        }
    }
</script>

<div id="layersPanel">
    {#each $layers as layer, index}
        <div
            class="layer"
            style='color:{layer.get('color')}'
            class:selected={layer == $activeLayer}
            on:click={() => {
                if ($activeLayer == layer) {
                    $activeLayerIdx = null    
                } else {
                    $activeLayerIdx = index
                }
            }}
        >
            <p>{layer.get('prompt')}</p>
            <p> - </p>
        </div>
    {/each}
    <button on:click={onClick}> +</button>
</div>

<style>
    #layersPanel {
        position: fixed;
        right: 0px;
        top: 100px;
        z-index: 2;
        display: flex;
        flex-direction: column;
        background: white;
        border-top-left-radius: 4px;
        border-bottom-left-radius: 4px;
        align-items: center;
    }
    .layer {
        display: flex;
        justify-content: space-evenly;
        width: 200px;
        cursor: pointer;
    }
    .layer.selected {
        background: #ffa50080;
    }
</style>
