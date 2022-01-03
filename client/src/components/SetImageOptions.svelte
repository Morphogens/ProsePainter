<script lang="ts">
    import type DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import { mode, canvasSize } from "@/stores";
    import { Mode } from "../types";
    import { loadImage, imageToCanvas, thumbnailCanvas } from "@/utils";
    import { tick } from "svelte";
    import { MAX_IMAGE_SIZE, DEFAULT_IMAGES } from "@/constants";
    export let mainCanvas: DrawCanvas;

    let width: number = 512;
    let height: number = 512;
    $: width = Math.min(width, MAX_IMAGE_SIZE);
    $: height = Math.min(height, MAX_IMAGE_SIZE);
    async function onFiles(event) {
        const { files } = event.target;
        if (files.length) {
            const url = URL.createObjectURL(files[0]);
            await setImageByURL(url);
        }
    }

    function setEmpty() {
        canvasSize.set([width, height]);
        const whiteCanvas = document.createElement('canvas')
        whiteCanvas.width = width
        whiteCanvas.height = height
        const ctx = whiteCanvas.getContext('2d')
        ctx.fillStyle = 'white'
        ctx.fillRect(0, 0, width, height)
        mainCanvas.set(whiteCanvas)
    }

    async function setImageByURL(imageUrl: string) {
        const image = await loadImage(imageUrl);
        const canvas = thumbnailCanvas(imageToCanvas(image), MAX_IMAGE_SIZE);
        canvasSize.set([canvas.width, canvas.height]);
        await tick(); // DrawCanvases get recreated.
        mainCanvas.set(canvas);
    }
</script>

{#if $mode == Mode.SetImage && mainCanvas}
    <p>Start from an empty canvas:</p>
    <div id="number-container">
        <input type="number" min="128" max="1024" bind:value={width} />
        <p>x</p>
        <input type="number" min="128" max="1024" bind:value={height} />
    </div>
    <button on:click={setEmpty}> Create</button>
    <!-- <Button on:click={setEmpty}>Create</Button> -->
    <br />
    <!-- <hr style='width:100%'> -->
    <p>Or, select one:</p>
    <div id="template-images">
        {#each DEFAULT_IMAGES as imageUrl}
            <img
                src={imageUrl}
                on:click={() => setImageByURL(imageUrl)}
                alt=""
            />
        {/each}
    </div>
    <br />
    <!-- <hr style='width:100%'> -->
    <!-- <p>Or:</p> -->
    <label class="custom-file-upload">
        <input type="file" accept=".jpg, .jpeg, .png" on:change={onFiles} />
        Upload Image
    </label>
    <br />
{/if}

<style>
    #number-container {
        display: flex;
        justify-content: center;
    }
    input[type="number"] {
        width: 50px;
        display: inline-block;
        margin: 4px;
    }
    #template-images {
        display: flex;
        flex-wrap: wrap;
    }
    #template-images img {
        width: 50%;
        cursor: pointer;
    }

    input[type="file"] {
        display: none;
    }
    .custom-file-upload {
        background-color: #f8f9fa;
        border: 1px solid #ccc;
        display: inline-block;
        padding: 6px 12px;
        cursor: pointer;
    }
</style>
