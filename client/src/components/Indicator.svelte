<script lang="ts">
    export let state = false;
</script>

<div class="w-2 h-2 rounded-full" class:bg-red-500={!state} class:bg-green-500={state} />
