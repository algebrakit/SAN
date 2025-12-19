import { Point, Stroke } from './types';
import { pointToSegmentDistance } from './geometry';
import { ERASER_RADIUS } from './config';

/**
 * Check if a point is within eraser radius of any part of a stroke
 */
export function isStrokeInEraserRadius(
  stroke: Stroke,
  point: Point,
  radius: number = ERASER_RADIUS
): boolean {
  // Check distance from point to each segment in the stroke
  for (let i = 0; i < stroke.points.length - 1; i++) {
    const distance = pointToSegmentDistance(point, stroke.points[i], stroke.points[i + 1]);
    if (distance <= radius) {
      return true;
    }
  }

  // Also check distance to individual points (for single-point strokes or endpoints)
  for (const strokePoint of stroke.points) {
    const dx = point.x - strokePoint.x;
    const dy = point.y - strokePoint.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance <= radius) {
      return true;
    }
  }

  return false;
}

/**
 * Detect which strokes are within eraser radius of a point
 */
export function detectErasableStrokes(
  strokes: Stroke[],
  point: Point,
  radius: number = ERASER_RADIUS
): number[] {
  const erasableIds: number[] = [];

  strokes.forEach(stroke => {
    if (isStrokeInEraserRadius(stroke, point, radius)) {
      erasableIds.push(stroke.id);
    }
  });

  return erasableIds;
}

/**
 * Get the default eraser radius
 */
export function getDefaultEraserRadius(): number {
  return ERASER_RADIUS;
}
