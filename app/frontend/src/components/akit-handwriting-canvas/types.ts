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
