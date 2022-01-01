<script lang="ts">
    import { onMount } from "svelte";

    export let show = true;

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
                ProsePainter is a new way to digitally paint! You
                create 'magic-ink' from any word or sentence and apply it to an image. It is
                powered by machine-learning.
            </p>

            <h4>Tutorial</h4>
            <p>(coming soon)</p>
            <!-- <video url=''></video> -->
            <p>
                ProsePainter is made by
                <a href="https://www.morphogen.io/">Morphogen</a>, the creators
                of
                <a href="https://artbreeder.com/">Artbreeder</a>. It is
                <a href="https://github.com/Morphogens/prosepainter"
                    >open sourced</a
                >.
            </p>
            <p />
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
        height: 512px;
        max-width: 80%;
        background-color: white;
        cursor: default;
        border: 1px solid #ccc;
        padding-left: 20px;
        padding-right: 20px;
    }
    h4,
    p {
        text-align: center;
    }
    a {
        color: black;
    }
</style>
