import {contours} from 'd3-contour'

export function imageContour(
    canvas: HTMLCanvasElement, ctx:CanvasRenderingContext2D
) {
    // contours
    const width = canvas.width
    const height = canvas.height
    const data4 = ctx.getImageData(0, 0, width, height).data
    // Check if the image uses transparency by the alpha of top-left pixel.
    if (data4[3] === 255) {
        // If no transparency return the square.
        return [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ]
    } else {
        // Collect the alpha pixels to use for marching squares.
        const alpha: number[] = []
        for (let i = 3; i < data4.length; i += 4) {
            alpha.push(data4[i])
        }
        // console.time('contours')
        return contours()
            .size([width, height])
            .thresholds([10])(alpha)[0]
            .coordinates.flat()
    }
}
