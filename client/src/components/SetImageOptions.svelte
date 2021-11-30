<script lang="ts">
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import { mode, canvasSize } from "@/stores";
    import { Mode } from "../types";
    import { loadImage } from "@/utils";
    import { tick } from "svelte";
    export let mainCanvas: DrawCanvas;
    import image0Url from '../assets/startImage0.jpeg'
    import image1Url from '../assets/startImage1.jpeg'
    import image2Url from '../assets/startImage2.jpeg'
    import image3Url from '../assets/startImage3.jpeg'

    let defaultImageUrls = [image0Url, image1Url, image2Url, image3Url]
    let width:number = 512
    let height:number = 512

    async function onFiles(event) {
        const { files } = event.target;
        if (files.length) {
            const url = URL.createObjectURL(files[0]);
            await setImageByURL(url)
        }
    }

    function setEmpty() {
        canvasSize.set([width, height]);
        const image = new Image(width, height)
        mainCanvas.set(image);
    }

    async function setImageByURL(imageUrl:string) {
        const image = await loadImage(imageUrl);
        canvasSize.set([image.width, image.height]);
        await tick(); // DrawCanvases get recreated.
        mainCanvas.set(image);
    }
</script>

{#if $mode == Mode.SetImage && mainCanvas}
    <p>Start from an empty canvas:</p>
    <div id='number-container'>
        <input type="number" min="128" max="10000" bind:value={width}>
        <p>x</p>
        <input type="number" min="128" max="100000" bind:value={height}/>
    </div>
    <button on:click={setEmpty}> Create</button>
    <br>
    <p>Or, select an default:</p>
    <div id='template-images'>
        {#each defaultImageUrls as imageUrl}
            <img src={imageUrl} on:click={() => setImageByURL(imageUrl)} alt=''>
            
        {/each}
    </div>
    <br>
    <p>Or, select any image file:</p>
    <input type="file" accept=".jpg, .jpeg, .png" on:change={onFiles} />
    <br>
{/if}

<style>
    #number-container   {
        display: flex;
        justify-content: center;
    }
    input[type='number'] {
        width: 40px;
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
</style>