<script lang="typescript">
import router from 'page'
import Drawing from './routes/Drawing.svelte'
import Points from './routes/Points.svelte'

let page
let params
router('/', () => (page = Drawing))
router('/points', () => (page = Points))
router.start()
</script>

<svelte:component this={page} {params} />
    