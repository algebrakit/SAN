/**
 * Configuration constants for the akit-handwriting-canvas component
 */

import { SymbolAdjustment } from "../../components";

// =============================================================================
// Canvas Dimensions
// =============================================================================

export const CANVAS_WIDTH = 2000;
export const CANVAS_VISIBLE_HEIGHT = 280;                             // What user sees
export const CANVAS_HEIGHT = CANVAS_VISIBLE_HEIGHT + 200;             // +100px top, +100px bottom for vertical scroll
export const SCALE_FACTOR = 2; // High-DPI scaling (2x resolution for smoother rendering)
export const PREPEND_AREA_WIDTH = 200; // Extra space on left for prepending symbols

// =============================================================================
// Confirm Button Positioning (Hysteresis)
// =============================================================================

// When repositioning, use large gaps to stay out of the way
export const BUTTON_GAP_X_LARGE = 80;
export const BUTTON_GAP_X_MIN = 30;    // Minimum gap before forcing reposition
export const BUTTON_GAP_X_MAX = 150;   // Maximum gap before forcing reposition (e.g., after erasing)
export const BUTTON_GAP_Y_LARGE = 40;  // Gap below baseline when repositioning
export const BUTTON_GAP_Y_MIN = 10;    // Minimum gap before forcing reposition

// =============================================================================
// Stroke Appearance
// =============================================================================

export const STROKE_COLOR = '#333390';        // Dark blue for pen-like appearance
export const BASE_STROKE_WIDTH = 3.2;
export const MIN_STROKE_WIDTH = 1.5;          // Minimum width when drawing fast
export const MAX_STROKE_WIDTH = 3.5;          // Maximum width when drawing slow
export const STROKE_SMOOTHING_FACTOR = 0.5;   // How much to smooth width changes (0-1)

// =============================================================================
// Grid
// =============================================================================

export const GRID_SPACING = 50;               // Grid spacing in pixels
export const GRID_COLOR = '#d8dde3';          // Subtle gray to match design

// =============================================================================
// Eraser
// =============================================================================

export const ERASER_RADIUS = 15;              // Eraser radius in pixels
export const HIGHLIGHT_COLOR = '#ff0000';     // Red for eraser highlight/stroke deletion

// =============================================================================
// Pan Gesture
// =============================================================================

export const PAN_COOLDOWN_MS = 200;           // Cooldown after panning before drawing resumes

// =============================================================================
// Vertical Scroll (limited)
// =============================================================================

export const VERTICAL_SCROLL_CELLS = 2;                               // Allow ±2 cells of vertical scroll
export const VERTICAL_SCROLL_MAX = GRID_SPACING * VERTICAL_SCROLL_CELLS;  // ±100px

// =============================================================================
// Scroll Tracking
// =============================================================================

export const SCROLL_END_DELAY_MS = 150;       // Time to wait before considering scroll stopped
export const MIN_VISIBLE_STROKE_MARGIN = 50;  // Pixels of stroke that must remain visible when scrolling

// =============================================================================
// Auto-Scroll (for small screens)
// =============================================================================

export const AUTO_SCROLL_MIN_GAP = 100;        // px from right edge to trigger auto-scroll
export const AUTO_SCROLL_PREFERRED_GAP = 200;  // px of space to create after scrolling
export const AUTO_SCROLL_DEBOUNCE_MS = 700;   // Wait before auto-scrolling after stroke ends

// =============================================================================
// Auto-Convert
// =============================================================================

export const AUTO_CONVERT_DEBOUNCE_MS = 1000; // Wait before auto-converting after stroke ends

// =============================================================================
// Symbol likelihood adjustments
// =============================================================================
export const SYMBOL_ADJUSTMENTS: SymbolAdjustment[] = [
  { symbol: 'X', offset: 'PENALIZE' }, // resembles x and multiplication
  { symbol: 's', offset: 'PENALIZE' }, // resembles 5
  { symbol: 'S', offset: 'PENALIZE' }, // resembles 5
  { symbol: 'P', offset: 'PENALIZE' }, // resembles p
  { symbol: 'b', offset: 'PENALIZE' }, // resembles 6
  { symbol: 'G', offset: 'PENALIZE' }, // resembles 6
  { symbol: 'B', offset: 'PENALIZE' }, // resembles 8
];

// =============================================================================
// API
// =============================================================================

// API base URL
// Use 'http://localhost:5001' for local development
// Use '' (empty string) for production (same-origin deployment)
export const API_BASE_URL = 'http://localhost:5001';
