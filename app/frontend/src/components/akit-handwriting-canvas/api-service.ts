import { Stroke, SymbolAdjustment } from './types';
import { API_BASE_URL } from './config';

export interface ConversionResult {
  latex: string;
}

/**
 * Convert strokes to the format expected by the API
 */
function formatStrokesForApi(strokes: Stroke[]): number[][][] {
  return strokes.map(stroke =>
    stroke.points.map(point => [point.x, point.y])
  );
}

/**
 * Convert strokes to LaTeX via the backend API
 * @param strokes - Array of strokes to convert
 * @param symbolAdjustments - Optional array of symbol probability adjustments
 */
export async function convertStrokes(
  strokes: Stroke[],
  symbolAdjustments: SymbolAdjustment[] = []
): Promise<ConversionResult> {
  const strokeData = formatStrokesForApi(strokes);

  console.log('Sending stroke data to URL:', `${API_BASE_URL}/convert`);

  const requestBody: { strokes: number[][][]; symbol_adjustments?: SymbolAdjustment[] } = {
    strokes: strokeData,
  };

  // Only include symbol_adjustments if there are any
  if (symbolAdjustments.length > 0) {
    requestBody.symbol_adjustments = symbolAdjustments;
  }

  const response = await fetch(`${API_BASE_URL}/convert`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

/**
 * Get the API base URL
 */
export function getApiUrl(): string {
  return API_BASE_URL;
}
