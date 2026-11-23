export interface Landmark {
    x: number;
    y: number;
    z: number;
}

export interface NormalisedHand {
    features: Float32Array;
    size: number;
}

export interface ModelMetadata {
    classes: string[];
    mean: number[];
    std: number[];
}

export interface Prediction {
    letter: string | null;
    confidence: number;
    valid: boolean;
}