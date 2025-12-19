import { Point, Stroke } from './types';

/**
 * Check if two line segments intersect using counter-clockwise orientation test
 */
export function lineSegmentsIntersect(p1: Point, p2: Point, p3: Point, p4: Point): boolean {
  const ccw = (A: Point, B: Point, C: Point): boolean => {
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
  };

  return ccw(p1, p3, p4) !== ccw(p2, p3, p4) && ccw(p1, p2, p3) !== ccw(p1, p2, p4);
}

/**
 * Count how many times two strokes (point arrays) intersect
 */
export function countIntersections(stroke1: Point[], stroke2: Point[]): number {
  let count = 0;

  for (let i = 0; i < stroke1.length - 1; i++) {
    for (let j = 0; j < stroke2.length - 1; j++) {
      if (lineSegmentsIntersect(
        stroke1[i], stroke1[i + 1],
        stroke2[j], stroke2[j + 1]
      )) {
        count++;
      }
    }
  }

  return count;
}

/**
 * Calculate distance from a point to a line segment
 */
export function pointToSegmentDistance(point: Point, segStart: Point, segEnd: Point): number {
  const A = point.x - segStart.x;
  const B = point.y - segStart.y;
  const C = segEnd.x - segStart.x;
  const D = segEnd.y - segStart.y;

  const dot = A * C + B * D;
  const lenSq = C * C + D * D;
  let param = -1;

  if (lenSq !== 0) {
    param = dot / lenSq;
  }

  let xx: number;
  let yy: number;

  if (param < 0) {
    xx = segStart.x;
    yy = segStart.y;
  } else if (param > 1) {
    xx = segEnd.x;
    yy = segEnd.y;
  } else {
    xx = segStart.x + param * C;
    yy = segStart.y + param * D;
  }

  const dx = point.x - xx;
  const dy = point.y - yy;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculate the bounding box of all strokes
 */
export function getStrokesBoundingBox(strokes: Stroke[]): { minX: number; minY: number; maxX: number; maxY: number } | null {
  if (strokes.length === 0) {
    return null;
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const stroke of strokes) {
    for (const point of stroke.points) {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
    }
  }

  return { minX, minY, maxX, maxY };
}

/**
 * Calculate the Y coordinate at a given percentile across all stroke points
 */
export function getYPercentile(strokes: Stroke[], percentile: number): number | null {
  if (strokes.length === 0) {
    return null;
  }

  // Collect all Y values from all strokes
  const yValues: number[] = [];
  for (const stroke of strokes) {
    for (const point of stroke.points) {
      yValues.push(point.y);
    }
  }

  // Sort ascending
  yValues.sort((a, b) => a - b);

  // Calculate percentile index
  const index = Math.floor((percentile / 100) * (yValues.length - 1));
  return yValues[index];
}
