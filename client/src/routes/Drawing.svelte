<script lang="ts">
    import InfiniteViewer from "@/drawing/svelte-infinite-viewer";
    import DrawCanvas from "@/drawing/DrawCanvas.svelte";
    import OptionPanel from "@/drawing/OptionPanel.svelte";
    import LayersPanel from "@/drawing/LayersPanel.svelte";
    import { socket, socketOpen } from "@/lib/socket";
    import { activeLayer, layers, undo, layerImages } from "@/drawing/stores";
    import { onMount, tick } from "svelte";
    import Indicator from "@/Indicator.svelte";
    import { debounce } from "lodash";

    let run = false;
    let previewCanvas: HTMLCanvasElement;
    let previewCanvasCtx: CanvasRenderingContext2D;

    const width = 1024;
    const height = 512;

    let backgroundImage = new Image()

    let showLayers = true;

    async function drawLayers(){
        await tick();
        if (!previewCanvasCtx) {
            return;
        }

        for (const layer of $layers.slice()) {
            const image = layerImages.get(layer);
            // console.log(layer == $activeLayer, layer, $activeLayer);
            if (image && layer != $activeLayer) {
                console.log("DRAWING LAYER")
                previewCanvasCtx.drawImage(image, 0, 0);
            }
        }
    }

    async function drawBackground() {
        await tick();
        if (!previewCanvasCtx) {
            return;
        }
        if (backgroundImage.src == ""){
            console.log("RESETTING BACKGROUND")
            previewCanvasCtx.fillStyle = 'white';
            previewCanvasCtx.fillRect(0, 0, width, height);
            backgroundImage.src = previewCanvas.toDataURL()
        } else {
            console.log("DRAWING BACKGROUND")
            previewCanvasCtx.drawImage(backgroundImage, 0, 0, width, height)
        }

    }

    $: if ($layers || $activeLayer || showLayers) {
        drawBackground();
        if (showLayers){
            drawLayers()
        }
    }

    function sendMessage(topic: string, data: any) {
        console.log("Sending", data);
        if ($socketOpen) {
            socket.send(JSON.stringify({ topic, data }));
        }
    }

    function startGeneration(){
        if ($activeLayer){
            sendMessage(
                "start-generation",
                {
                    ...$activeLayer.toJSON(),
                    ...{'backgroundImg': backgroundImage.src},
                },
                // $layers.map((l) => l.toJSON())
            );
            run = true
            $activeLayer.set(null)
        } else {
            alert("AT LEAST ONE LAYER SHOULD BE SELECTED!")
            run = false
        }
    }

    function stopGeneration(){
        sendMessage("stop-generation", {});
        run = false
    }

    let img: string | undefined = undefined;
    socket.addEventListener("message", (e) => {
        console.log("MESSAGE RECEIVED!")
        
        const message = JSON.parse(e.data)

        if (message.image) {
            console.log("IMAGE RECEIVED!")
            img = "data:text/plain;base64," + message.image;
            backgroundImage.src = img; 

            drawBackground()

            // let etaieagenew Image()
            // image.src = img

            // for (const layer of $layers.slice().reverse()) {
            //     layerImages.set(layer, image);
            // }
        } else{
            console.log("NO IMAGE RECEIVED!")
        }
    });

    function onKeyDown(e: KeyboardEvent) {
        if (e.code === "KeyZ" && (e.metaKey === true || e.ctrlKey === true)) {
            if (e.shiftKey) {
                undo.redo();
            } else {
                undo.undo();
            }
        }
    }

    function onKeyUp(e: KeyboardEvent) {}

    onMount(() => {
        previewCanvasCtx = previewCanvas.getContext("2d");
    });
    $: console.log($activeLayer);
    
</script>

<svelte:window on:keydown={onKeyDown} on:keyup={onKeyUp} />
<div class="flex items-center px-2 py-1">
    <Indicator state={$socketOpen} />
    <div class="ml-1">Connection is {$socketOpen ? "open" : "closed"}</div>
</div>
{#if $activeLayer}
    <OptionPanel />
{/if}
<LayersPanel />
<InfiniteViewer
    className="viewer"
    usePinch={true}
    rangeX={[-256, 256]}
    rangeY={[-256, 256]}
>
    <div class="viewport" style="width:{width}px">
        <div id="content" style="width:{width}px;height:{height}px">
            <div style='border-right:1px solid;width:{width}px'>
                <canvas
                    id="previewCanvas"
                    bind:this={previewCanvas}
                    {width}
                    {height}
                    style="opacity:{$activeLayer || showLayers ? 0.5 : 1.};"
                />
                {#if $activeLayer}
                    <DrawCanvas {width} {height} />
                {/if}
            </div>
            <!-- <div>
                {#if img}
                    <img
                        id="previewImage"
                        class="select-none h-64 w-64 absolute"
                        draggable="false"
                        src={img}
                        alt="previewImage"
                        style="width:{width}px;height:{height}px"
                    />
                {/if}
            </div> -->

        </div>
        <div id="canvasButtons">
            {#if run}
                <button on:click={() => (stopGeneration())}> Stop </button>
            {:else if $socketOpen}
                <button on:click={() => (startGeneration())}> Start </button>
            {:else}
                <button on:click={() => (alert("SOCKET NOT CONNECTED :("))}> Start </button>
                <!-- <p> Socket not open </p> -->
            {/if}
            {#if showLayers}
                <button on:click={() => (showLayers=false)}> Hide Layers </button>
            {:else if $socketOpen}
                <button on:click={() => (showLayers=true)}> Show Layers </button>
            {/if}
            <!-- <button on:click={() => sendMessage("state", { reset: true })}>
                Reset
            </button> -->
        </div>
    </div>
</InfiniteViewer>

<style>
    #previewCanvas {
        top: 0px;
        position: absolute;
        pointer-events: none;
    }
    #content {
        display: flex;
        border-bottom: 1px dashed;
    }
    #canvasButtons {
        display: flex;
        justify-content: space-between;
    }
    :global(.viewer) {
        border: 1px solid black;
        position: relative;
        width: 100vw;
        height: 100vh;
        background: gray;
    }
    .viewport {
        position: relative;
        margin: 100px;
        background: white;
        box-shadow: -1px 4px 8px 0px rgba(0, 0, 0, 0.61);
    }
    .hidden {
        display: none;
    }
</style>
