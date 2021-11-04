<script lang="ts">
    import DrawCanvas from '@/drawing/DrawCanvas.svelte'
    import { learningRate, prompt } from '../stores'
    export let drawCanvas: DrawCanvas

</script>

<div id="optionPanel">
    {#if drawCanvas}
    <input type="text" id="lname" name="lname" bind:value={$prompt}>
    <button on:click={() => (drawCanvas.erasing = false)} class:selected={!drawCanvas.erasing}>
        <img src="/pencil.svg" />
    </button>
    <button on:click={() => (drawCanvas.erasing = true)} class:selected={drawCanvas.erasing}>
        <img src="/eraser.png" />
    </button>
    
    
    <button on:click={() => drawCanvas.clear()}>
        <p> Clear Mask </p>
    </button>
    <p> Radius={drawCanvas.radius} </p>
    <input type="range" bind:value={drawCanvas.radius} min=1 max=96/>
    <p> Softness </p>
    <input type="range" bind:value={drawCanvas.softness} min=0 max=20 step="any"/>
    <p> learningRate </p>
    <input type="range" bind:value={$learningRate} min=0 max=500.0 step=1/>
    {$learningRate / 1000}
    {/if}
    
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
