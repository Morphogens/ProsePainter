
export enum Mode {
    SetImage,
    DirectDraw,
    MaskDraw,
    Optimizing,
    PausedOptimizing,
}

export interface OptimizationResult {
    step: number,
    num_iterations: number,
    image: HTMLImageElement
}