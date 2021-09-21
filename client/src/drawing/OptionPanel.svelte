<script lang="ts">
    import { erasing, radius } from "./stores"
</script>

<div id="optionPanel">
    <button on:click={() => ($erasing = false)} class:selected={!$erasing}>
        <img src="/pencil.svg" />
    </button>
    <button on:click={() => ($erasing = true)} class:selected={$erasing}>
        <img src="/eraser.png" />
    </button>
    <input type="range" orient="vertical" bind:value={$radius} min=1 max=96/>
</div>

<style>
    #optionPanel {
        position: fixed;
        left: 0px;
        top: 100px;
        z-index: 2;
        display: flex;
        flex-direction: column;
        background: white;
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
        align-items: center;
    }
    button > img {
        width: 50px;
        height: 50px;
    }
    button.selected {
        background: #ffa50080;
    }
    input[type="range"][orient="vertical"] {
        writing-mode: bt-lr; /* IE */
        -webkit-appearance: slider-vertical; /* WebKit */
        width: 8px;
        height: 100px;
        padding: 0 5px;
    }
</style>
