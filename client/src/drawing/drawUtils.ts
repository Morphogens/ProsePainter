
export function drawCircle(
    ctx,
    x,
    y,
    r,
    fill = true,
    stroke = false,
    clip = false
) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI, false);
    if (clip) {
        ctx.clip();
    }
    if (stroke) {
        ctx.stroke();
    }
    if (fill) {
        ctx.fill();
    }
    ctx.closePath();
}


export function drawLine(
    ctx:CanvasRenderingContext2D,
    x1:number,
    y1:number,
    x2:number,
    y2:number,
    radius:number
) {
    ctx.lineWidth = radius
    ctx.beginPath();
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()
}

export function drawLines(
    ctx:CanvasRenderingContext2D,
    points: number[][]
) {
    if (points.length < 2) {
        return
    }
    ctx.beginPath()
    ctx.moveTo(points[0][0], points[0][1])
    for (let idx = 1; idx < points.length; idx++) {
        ctx.lineTo(points[idx][0], points[idx][1])
    }
    ctx.stroke()
}