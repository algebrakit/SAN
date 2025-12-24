import { SymbolAdjustment } from '../akit-config-handwriting/types';

export { SymbolAdjustment };

export interface Point {
  x: number;
  y: number;
  timestamp?: number;  // For speed calculation
  width?: number;      // Calculated stroke width at this point
}

export interface Stroke {
  points: Point[];
  id: number;
}

export interface HandwritingCanvasState {
  strokes: Stroke[];
  strokeCounter: number;
  scrollLeft: number;
}
