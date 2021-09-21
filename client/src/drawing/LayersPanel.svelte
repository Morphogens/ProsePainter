<script lang="ts">
    import {
        layers,
        activeLayer,
        activeLayerIdx,
        addLayer,
        removeLayer,
    } from "./stores";

    function onClick() {
        const userPrompt = prompt("Enter a prompt");
        if (userPrompt) {
            addLayer(userPrompt);
        }
    }
</script>

<div id="layersPanel">
    {#each $layers as layer, index}
        <div class="layer" class:selected={layer == $activeLayer}>
            <div
                class="labelName"
                style="color:{layer.get('color')}"
                on:click={() => {
                    if ($activeLayer == layer) {
                        $activeLayerIdx = null;
                    } else {
                        $activeLayerIdx = index;
                    }
                }}
            >
                <p>{layer.get("prompt")}</p>
            </div>

            <div
                class="labelDelete"
                on:click={() => removeLayer(layer)}
            >
                <p>-</p>
            </div>
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
    .labelName {
        width: 80%;
    }
    .labelDelete {
        width: 20%;
    }
    .layer.selected {
        background: #ffa50080;
    }
</style>
