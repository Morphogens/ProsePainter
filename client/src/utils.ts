export async function loadImage(src: string): Promise<HTMLImageElement> {
    return await new Promise((resolve, reject) => {
        const image = new Image()
        image.src = src
        image.onload = () => resolve(image)
        image.onerror = () => reject(new Error('could not load image'))
    })
}

export function imageToCanvas(img: HTMLImageElement): HTMLCanvasElement {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D
    canvas.height = img.naturalHeight
    canvas.width = img.naturalWidth
    ctx.drawImage(img, 0, 0)
    return canvas
}

export function imgTob64(img: HTMLImageElement): string {
    const canvas = imageToCanvas(img)
    return canvas.toDataURL()
}

export function downloadCanvas(
    canvas: HTMLCanvasElement,
    filename: string = 'download.png'
) {
    const url = canvas.toDataURL()
    const aDownloadLink = document.createElement('a')
    aDownloadLink.download = filename
    aDownloadLink.href = url
    aDownloadLink.click()
}


export function mergeCanvas(
    canvas1: HTMLCanvasElement,
    canvas2: HTMLCanvasElement
): HTMLCanvasElement {
    const newCanvas = document.createElement('canvas')
    const newCtx = newCanvas.getContext('2d')
    newCanvas.width = canvas1.width
    newCanvas.height = canvas1.height
    newCtx.drawImage(canvas1, 0, 0)
    newCtx.drawImage(canvas2, 0, 0)
    return newCanvas
}

export function thumbnailCanvas(
    canvas: HTMLCanvasElement,
    maxSize: number
): HTMLCanvasElement {
    const scale = Math.min(1, maxSize / Math.max(canvas.height, canvas.width))
    const thumbnail = document.createElement('canvas')
    const thumbnailCtx = thumbnail.getContext('2d')
    thumbnail.width = Math.round(scale * canvas.width)
    thumbnail.height = Math.round(scale * canvas.height)
    thumbnailCtx.drawImage(
        canvas,
        0, 0,
        canvas.width, canvas.height,
        0, 0,
        thumbnail.width,
        thumbnail.height
    )
    return thumbnail
}