export async function loadImage(src: string): Promise<HTMLImageElement> {
    return await new Promise((resolve, reject) => {
        const image = new Image()
        image.src = src
        image.onload = () => resolve(image)
        image.onerror = () => reject(new Error('could not load image'))
    })
}

function imageToCanvas(img: HTMLImageElement): HTMLCanvasElement {
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

