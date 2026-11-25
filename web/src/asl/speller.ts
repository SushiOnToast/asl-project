// replicates the buffer and fingerspeller logic from python script

// manual implementation of the deque(suitable for the fingerspeller nothing more)
export class ProbabilityBuffer {
  private items: Float32Array[] = [];
  private readonly maxLen: number;

  constructor(maxLen = 8) {
    this.maxLen = maxLen;
  }

  push(probs: Float32Array): void {
    this.items.push(probs);
    if (this.items.length > this.maxLen) {
      this.items.shift(); // removes first off queue
    }
  }

  mean(): Float32Array | null {
    if (this.items.length === 0) return null;
    const out = new Float32Array(this.items[0].length);
    for (const p of this.items) {
      for (let i = 0; i < out.length; i++) out[i] += p[i];
    }
    for (let i = 0; i < out.length; i++) out[i] /= this.items.length;
    return out;
  }

  clear(): void {
    this.items = [];
  }
}

export class FingerSpeller {
  private currLetter: string | null = null;
  private letterFrames = 0;
  private locked = false;

  private readonly confidenceThreshold: number;
  private readonly holdFrames: number;

  constructor(confidenceThreshold = 0.8, holdFrames = 20) {
    this.confidenceThreshold = confidenceThreshold;
    this.holdFrames = holdFrames;
  }

  update(letter: string | null, conf: number): string | null {
    if (letter === null || conf < this.confidenceThreshold) {
      this.reset();
      return null;
    }

    if (letter !== this.currLetter) {
      this.currLetter = letter;
      this.letterFrames = 1;
      this.locked = false;
    } else {
      this.letterFrames++;
    }

    if (this.letterFrames >= this.holdFrames && !this.locked) {
      this.locked = true;
      return letter;
    }

    return null;
  }

  get progress(): number {
    return Math.min(this.letterFrames / this.holdFrames, 1.0);
  }

  reset(): void {
    this.currLetter = null;
    this.letterFrames = 0;
    this.locked = false;
  }
}
