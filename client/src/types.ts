
export enum Mode {
    SetImage,
    DirectDraw,
    MaskDraw,
    Optimizing,
    PausedOptimizing,
}

export interface OptimizationResults {
    // step: number,
    num_iterations: number, // The total possible.
    images: HTMLImageElement[]
}