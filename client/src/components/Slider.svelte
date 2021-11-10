<script lang="ts">
    export let name: string;
    export let val: number;
    export let min: number = 0;
    export let max: number = 1;
    export let step: number | string = "any";
</script>
<p>{name} = {val}</p>
<input type="range" bind:value={val} {min} {max} {step} />

<style>
    p {
        margin-bottom: 2px;
    }
</style>