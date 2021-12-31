<script lang="ts">
    export let name: string;
    export let val: number;
    export let min: number = 0;
    export let max: number = 1;
    export let step: number | string = "any";
</script>

<div class="slider-container">
    <!-- <p>{name}: {val}</p> -->
    <div class="textbox">
        <p style="float:left">{name}:</p>
        <p style="float:right">{val}</p>
        <div style="clear: both;"></div>
    </div>
    <input type="range"  class="slider" bind:value={val} {min} {max} {step} />
</div>

<style>
    .textbox p {
        margin-bottom: 0px !important;
        font-size: 14px;
    }
    .slider-container {
        /* padding: 4px; */
    }
    .slider {
        -webkit-appearance: none;
        width: 100%;
        height: 24px;
        border-radius: 12px;
        background: #d3d3d3;
        outline: none;
        opacity: 0.7;
        -webkit-transition: 0.2s;
        transition: opacity 0.2s;
        cursor: pointer;
    }

    .slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #04aa6d;
        cursor: ew-resize;
    }

    .slider::-moz-range-thumb {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #04aa6d;
        cursor: ew-resize;
    }
    .textbox {
        user-select: none;
    }
    /* p {
        text
    } */
</style>
