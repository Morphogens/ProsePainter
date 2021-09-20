<script lang="ts">
    export let text: string;
    export let pos: [number, number];
    export let radius: number;
</script>



<table>
    <tr>
        <td class="p-4">Text</td>
        <td><input class="p-4 border rounded w-72" bind:value={text} /></td>
    </tr>
    <tr>
        <td class="p-4">Radius</td>
        <td>
            <input type=number bind:value={radius} min=0 max=1 step=0.1>
            <input type=range bind:value={radius} min=0 max=1 step=0.1>
        </td>
    <tr>
        <td class="p-4">Position</td>
        <td>
            <div class="ml-16 border-l">
                <div>
                    X
                    <input type=number bind:value={pos[0]} min=0 max=1 step=0.1>
                    <input type=range bind:value={pos[0]} min=0 max=1 step=0.1>
                </div>
                <div>
                    Y
                    <input type=number bind:value={pos[1]} min=0 max=1 step=0.1>
                    <input type=range bind:value={pos[1]} min=0 max=1 step=0.1>
                </div>
            </div>
        </td>
    </tr>

</table>