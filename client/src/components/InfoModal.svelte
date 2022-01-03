<script lang="ts">
    import { onMount } from "svelte";

    export let show = false;

    export function toggle() {
        show = !show;
    }
    onMount(() => {
        const hasOpened = localStorage.getItem("hasOpened");
        if (hasOpened == null) {
            show = true;
            localStorage.setItem("hasOpened", "1");
        }
    });
</script>

{#if show}
    <div class="modal-background" on:click|self={() => (show = false)}>
        <div class="modal-content">
            <h4>ProsePainter-<i>Beta</i></h4>
            <p>
                ProsePainter is a new way to digitally paint! You create
                'magic-ink' from any word or sentence and apply it to an image.
            </p>
            
            <!-- <hr> -->
            <hr>
            <!-- <h4>Tutorial</h4> -->
            <p>The magic-ink transforms images where it is applied, use it to modify an image or to create one from sketches.</p>
            <p style='margin-bottom:0px;'>Example: applying the ink <i>"colorful flowers"</i>.</p>
            <div class='image-container'>
                <div>
                    <p>Start Image</p>
                    <img src='/images/modal/before.jpg'>
                </div>
                <div>
                    <p>Ink Mask</p>
                    <img src='/images/modal/mask.jpg'>
                </div>
                <div>
                    <p>Result</p>
                    <img src='/images/modal/after.jpg'>
                </div>
            </div>
            <!-- <p>1. First, select a staring image or upload your own.</p>
            <p>2. Try creating </p> -->
            <!-- <video url=''></video> -->
            <!-- <p /> -->
            <br>
            <hr>
            <p>
                ProsePainter is made by
                <a href="https://www.morphogen.io/">Morphogen</a>, the creators
                of
                <a href="https://artbreeder.com/">Artbreeder</a>. It is powered by machine-learning and is 
                <a href="https://github.com/Morphogens/prosepainter"
                    >open sourced</a
                >.
            </p>
        </div>
    </div>
{/if}

<style>
    .modal-background {
        position: fixed;
        z-index: 99;
        top: 0px;
        left: 0px;
        width: 100vw;
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: rgba(169, 169, 169, 0.336);
        backdrop-filter: blur(2px);
        cursor: pointer;
    }
    .modal-content {
        width: 512px;
        min-height: 512px;
        max-width: 80%;
        background-color: white;
        cursor: default;
        border: 1px solid #ccc;
        padding-left: 30px;
        padding-right: 30px;
    }
    h4,
    p {
        text-align: center;
    }
    a {
        color: black;
    }

    .image-container {
        display: flex;
        justify-content: center;
    }
    .image-container > * {
        display: flex;
        flex-direction: column;
        width: 33%;
    }
    .image-container img {
        width: 100%;
    }
    .image-container p  {
        font-weight: bold;
    }
</style>
