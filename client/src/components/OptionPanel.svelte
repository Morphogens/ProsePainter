<script lang="ts">
    import { erasing, radius, softness, opacity, clear } from "../drawing/stores"
</script>

<div id="optionPanel">
    <input type="text" id="lname" name="lname" value="a dog">
    <button on:click={() => ($erasing = false)} class:selected={!$erasing}>
        <img src="/pencil.svg" />
    </button>
    <button on:click={() => ($erasing = true)} class:selected={$erasing}>
        <img src="/eraser.png" />
    </button>
    
    <button on:click={() => clear()}>
        <p> Clear Mask </p>
    </button>
    <p> Radius={$radius} </p>
    <input type="range" bind:value={$radius} min=1 max=96/>
    <!-- <p> Opacity </p>
    <input type="range" bind:value={$opacity} min=0.0 max=1.0 step="any"/> -->
    <p> Softness </p>
    <input type="range" bind:value={$softness} min=0 max=20 step="any"/>
    
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
        border: 1px solid;
    }
    button {
        cursor: pointer;
    }
    button > img {
        width: 50px;
        height: 50px;
    }
    button.selected {
        background: #ffa50080;
    }
    input{
        width: 96px;
    }
    input[type="range"][orient="vertical"] {
        writing-mode: bt-lr; /* IE */
        -webkit-appearance: slider-vertical; /* WebKit */
        width: 8px;
        height: 100px;
        padding: 0 5px;
    }
</style>
