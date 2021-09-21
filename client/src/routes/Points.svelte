<script lang="ts">
  import Indicator from '../Indicator.svelte';
  import { socket, socketOpen } from '../lib/socket';
  import { scaleLinear } from 'd3';
  import { debounce } from 'lodash-es';
  import * as knobby from 'svelte-knobby';
  import '../TailwindCSS.svelte';
  import Moveable from "svelte-moveable";
import AnchorEditor from '../AnchorEditor.svelte';

  interface Anchor {
    text: string;
    id: string;
    pos: [number, number];
    radius: number;
  }

  let anchors: Anchor[] = [
    {
      text: 'a red circle',
      id: '0',
      pos: [0.1, 0.1],
      radius: 0.1,
    },
    {
      text: 'a green rectangle',
      id: '1',
      pos: [0.9, 0.9],
      radius: 0.1,
    },
  ];


  let activeAnchorIdx = 0;
  let activeAnchor = anchors[0];

  $: updateAnchor(activeAnchor)

  function updateAnchor(anchor) {
    if (!anchor) return;
    anchors = anchors.map(p => p.id === anchor.id ? anchor : p);
  }

  let img = undefined;
  socket.addEventListener('message', (e) => {
    const message = JSON.parse(e.data);
    if (message.image) {
      img = 'data:text/plain;base64,' + message.image;
    }
  });

  $: scaleX = scaleLinear().domain([0, 1]).range([0, imageWidth]);
  $: scaleY = scaleLinear().domain([0, 1]).range([0, imageHeight]);
  let imageHeight: number;
  let imageWidth: number;

  let coord = { x: 0, y: 0 };
  function onMouseMove(e: MouseEvent) {
    coord = {
      x: scaleX.invert(e.offsetX),
      y: scaleY.invert(e.offsetY),
    };
  }


  function sendMessage(topic: string, data: any) {
    if ($socketOpen) {
      socket.send(JSON.stringify({ topic, data }));
    }
  }

  function _sendState(state) {
    sendMessage('state', state);
  }

  const submitState = debounce(_sendState, 100);
  $: submitState({anchors, run});

  function switchPromptClicked() {
    activeAnchorIdx = (activeAnchorIdx + 1) % anchors.length;
    activeAnchor = anchors[activeAnchorIdx];
  }

  function onImageClicked() {
    // $options.anchor.pos = coord;   
  }

  let run = false;
  let containerElement: HTMLElement;
  let moveableElement: Moveable;

  function anchorClicked(e: MouseEvent, anchor: Anchor) {
    const m = moveableElement.getInstance();
    const target = e.target as HTMLElement;
    m.target = target
    activeAnchor = anchor;
  }


	function onDragStart(e) {
		// const target = e.target;
		// const key = target.dataset.key;
		// const frame = entityFrames[key];
		e.set([scaleX(activeAnchor.pos[0]), scaleY(activeAnchor.pos[1])]);
	}
	function onDrag({ target, beforeTranslate }) {
		activeAnchor.pos[0] = scaleX.invert(beforeTranslate[0]);
		activeAnchor.pos[1] = scaleY.invert(beforeTranslate[1]);
  }
</script>



<button on:click={switchPromptClicked} class="p-4 rounded border m-2">Switch Prompt</button>

<main>
  <div class="flex items-center">
    <Indicator state={$socketOpen} />
    <div class="ml-1">Connection is {$socketOpen ? 'open' : 'closed'}</div>
  </div>
  <div class="flex flex-col w-72">
      <button 
      class="w-24 text-white mb-4 transition-colors p-2 rounded"
      class:bg-red-500={run}
      class:bg-green-500={!run}
      on:click={() => run = !run}>
        {run ? "Stop" : "Run"}</button>
    <AnchorEditor bind:text={activeAnchor.text} bind:pos={activeAnchor.pos} bind:radius={activeAnchor.radius} />
  </div>
  <div>{coord.x.toPrecision(2)}%x{coord.y.toPrecision(2)}%</div>
  <div
    class="bg-gray-100 inline-block relative"
    bind:this={containerElement}
    bind:clientHeight={imageHeight}
    bind:clientWidth={imageWidth}
    on:mousemove={onMouseMove}
    on:click={onImageClicked}
  >
    <img class="select-none" draggable="false" src={img} />
    {#each anchors as anchor}
    <div
    class="absolute w-4 h-4 bg-gray-700 border border-white rounded-full"
    style="left: {(anchor.pos[0] - anchor.radius / 2) * 100}%; top: {(anchor.pos[1] - anchor.radius/2) * 100}%; width: {anchor.radius * 100}%; height: {anchor.radius * 100}%;"
    on:mousedown={(e) => anchorClicked(e, anchor)}
    ></div>
    {/each}
  </div>
  {#if containerElement}
  <Moveable
    bind:this={moveableElement}
    container={containerElement}
    draggable={true}
    on:dragStart={({ detail: e }) => onDragStart(e)}
    on:drag={({ detail: e }) => onDrag(e)}
  />
  {/if}
</main>

<style>
  :root {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans',
      'Helvetica Neue', sans-serif;
  }

  main {
    text-align: center;
    padding: 1em;
    margin: 0 auto;
  }

  img {
    height: 16rem;
    width: 16rem;
  }

  h1 {
    color: #ff3e00;
    text-transform: uppercase;
    font-size: 4rem;
    font-weight: 100;
    line-height: 1.1;
    margin: 2rem auto;
    max-width: 14rem;
  }

  p {
    max-width: 14rem;
    margin: 1rem auto;
    line-height: 1.35;
  }

  @media (min-width: 480px) {
    h1 {
      max-width: none;
    }

    p {
      max-width: none;
    }
  }
</style>
