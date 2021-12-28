// import { imageContour } from "./imageContour";

export function drawGrid(
    canvasSize: number[],
    gridSize: number,
    colorA: string = 'lightblue',
    colorB: string = 'yellow'
): [HTMLCanvasElement, CanvasRenderingContext2D] {
    const canvas: HTMLCanvasElement = document.createElement('canvas')
    canvas.width = canvasSize[0]
    canvas.height = canvasSize[1]
    const ctx = canvas.getContext('2d')
    const numCols = Math.round(canvasSize[0] / gridSize)
    const numRows = Math.round(canvasSize[1] / gridSize)
    for (let row = 0; row < numRows; row += 1) {
        for (let col = 0; col < numCols; col += 1) {
            ctx.beginPath()
            ctx.fillStyle = row % 2 == col % 2 ? colorA : colorB
            const x = col * gridSize
            const y = row * gridSize
            ctx.rect(x, y, gridSize, gridSize)
            ctx.fill()
        }
    }
    return [canvas, ctx]
}

export function drawMaskGridAlpha(
    dstCtx: CanvasRenderingContext2D,
    canvasSize: number[],
    maskCanvas: HTMLCanvasElement,
    gridCanvas: HTMLCanvasElement
) {
    dstCtx.clearRect(0, 0, canvasSize[0], canvasSize[1]);
    dstCtx.globalAlpha = 0.6
    dstCtx.drawImage(gridCanvas, 0, 0)
    dstCtx.globalAlpha = 1.0
    dstCtx.globalCompositeOperation = "destination-in";
    dstCtx.drawImage(maskCanvas, 0, 0)
    dstCtx.globalCompositeOperation = "source-over"; // reset to defauly.
}

// function drawContours(contours: number[][][]) {
//     outlineCtx.clearRect(0, 0, $canvasSize[0], $canvasSize[1]);
//     outlineCtx.lineWidth = 1;
//     outlineCtx.setLineDash([3, 3]);
//     for (const poly of contours) {
//         drawLines(outlineCtx, poly);
//     }
// }

// function drawGrid(gridSize:number): [HTMLCanvasElement, CanvasRenderingContext2D] {
//     const canvas:HTMLCanvasElement = document.createElement('canvas')
//     canvas.width = $canvasSize[0]
//     canvas.height = $canvasSize[1]
//     const ctx = canvas.getContext('2d')
//     ctx.fillStyle = "white";
//     ctx.beginPath();
//     for (let x = 0; x < $canvasSize[0]; x += 2 * gridSize) {
//         for (let row = 0; row < $canvasSize[0] / gridSize; row += 1) {
//             const y = row * gridSize;
//             const xOffset = row % 2 == 0 ? 0 : gridSize;
//             ctx.rect(x + xOffset, y, gridSize, gridSize);
//         }
//     }
//     ctx.fill();
//     return [canvas, ctx]
// }


// function drawMaskGridInverted(maskCanvas: HTMLCanvasElement, gridSize: number =3) {
//     outlineCtx.clearRect(0, 0, $canvasSize[0], $canvasSize[1]);
//     // First union the grid and mask
//     if ($mainCanvas) {
//         var imageData = $mainCanvas.getContext().getImageData(0, 0, $canvasSize[0], $canvasSize[1]);
//         var data = imageData.data;
//         for (let i = 0; i < data.length; i+=4) {
//             data[i] = Math.min(255, 255 - data[i] + 40)
//             data[i+1] = 255 - data[i+1]
//             data[i+2] = 255 - data[i+2]
//         }            
//         outlineCtx.putImageData(imageData, 0, 0)
//     }
//     outlineCtx.globalCompositeOperation = "destination-in";
//     outlineCtx.drawImage(maskCanvas, 0, 0);        
//     const gridCanvas = drawGrid(gridSize)[0]
//     outlineCtx.drawImage(gridCanvas, 0, 0);
// }

// function drawMaskGridBinary(maskCanvas: HTMLCanvasElement, gridSize: number =3) {
//     outlineCtx.clearRect(0, 0, $canvasSize[0], $canvasSize[1]);
//     // First union the grid and mask
//     if ($mainCanvas) {
//         const smallCanvas:HTMLCanvasElement = document.createElement('canvas')
//         const smallCtx = smallCanvas.getContext('2d')
//         smallCanvas.width = Math.round($canvasSize[0] / gridSize)
//         smallCanvas.height = Math.round($canvasSize[1] / gridSize)
//         smallCtx.drawImage(
//             $mainCanvas.getCanvas(),
//             0, 0,
//             $canvasSize[0],
//             $canvasSize[1],
//             0, 0,
//             smallCanvas.width, 
//             smallCanvas.height, 
//         )
//         const imageData = smallCtx.getImageData(0, 0, smallCanvas.width, smallCanvas.height)
//         const data = imageData.data
//         console.log(data.length);

//         // ctx.fillStyle = "white";
//         const numCols:number = Math.round($canvasSize[0] / gridSize)
//         const numRows:number = Math.round($canvasSize[1] / gridSize)

//         // for (let x = 0; x < $canvasSize[0]; x += 2 * gridSize) {
//         console.time('draw grid')
//         for (let row = 0; row < numRows; row += 1) {
//             for (let col = 0; col < numCols; col += 1) {
//                 if (row %2 != col%2) {
//                     continue
//                 }
//                 outlineCtx.beginPath()
//                 const x:number = col * gridSize
//                 const y:number = row * gridSize
//                 const i = 4 * ((row*numRows)+col)
//                 const [r, g, b] = [data[i], data[i+1], data[i+2]]                    
//                 const lumin = r*0.299 + g*0.587 + b*0.114
//                 // outlineCtx.fillStyle = `rgb(${r}, ${g}, ${b})`
//                 // outlineCtx.fillStyle = `rgb(${r}, ${g}, ${b})`
//                 // outlineCtx.fillStyle = lumin > 186 ? 'black' : 'white'
//                 // outlineCtx.fillStyle = lumin > 210 ? 'black' : 'white'
//                 // outlineCtx.fillStyle = lumin > 210 ? 'lightgray' : 'darkgray'
//                 outlineCtx.rect(x, y, gridSize, gridSize);
//                 outlineCtx.fill();
//             }
//         }
//         console.timeEnd('draw grid')
//     }
//     outlineCtx.globalCompositeOperation = "destination-in";
//     outlineCtx.drawImage(maskCanvas, 0, 0);        
//     // const gridCanvas = drawGrid(gridSize)[0]
//     // outlineCtx.drawImage(gridCanvas, 0, 0);
// }