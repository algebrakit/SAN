import { Stroke } from './types';
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
 */
export async function convertStrokes(strokes: Stroke[]): Promise<ConversionResult> {
  const strokeData = formatStrokesForApi(strokes);

  console.log('Sending stroke data to URL:', `${API_BASE_URL}/convert`);

  const response = await fetch(`${API_BASE_URL}/convert`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      strokes: strokeData
    })
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
