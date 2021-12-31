import { contours } from 'd3-contour'


function _imageContour(
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D
): number[][][] {
    const width = canvas.width
    const height = canvas.height
    const data4 = ctx.getImageData(0, 0, width, height).data
    // Check if the image uses transparency by the alpha of top-left pixel.
    if (data4[3] === 255) {
        // If no transparency return the square.
        return [[
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ]]
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

const contourCanvas = document.createElement('canvas')
contourCanvas.width = 300
contourCanvas.height = 300
const contourCtx = contourCanvas.getContext('2d')

export function imageContour(maskCanvas: HTMLCanvasElement) {
    // A wrapper for the contours that places it in a smaller canvas first for optimization.
    contourCtx.clearRect(0, 0, contourCanvas.width, contourCanvas.height)
    contourCtx.drawImage(
        maskCanvas,
        0, 0,
        maskCanvas.width,
        maskCanvas.height,
        0, 0,
        contourCanvas.width,
        contourCanvas.height,
    )
    const scaleX = maskCanvas.width / contourCanvas.width
    const scaleY = maskCanvas.height / contourCanvas.height
    return _imageContour(contourCanvas, contourCtx).map(poly => (
        poly.map(p => [p[0] * scaleX, p[1] * scaleY])
    ))
}