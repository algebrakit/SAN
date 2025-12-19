import { Point, Stroke } from './types';
import {
  STROKE_COLOR,
  BASE_STROKE_WIDTH,
  MIN_STROKE_WIDTH,
  MAX_STROKE_WIDTH,
  STROKE_SMOOTHING_FACTOR,
  GRID_SPACING,
  GRID_COLOR,
  HIGHLIGHT_COLOR
} from './config';

export interface RendererConfig {
  strokeColor?: string;
  baseStrokeWidth?: number;
  minStrokeWidth?: number;
  maxStrokeWidth?: number;
  smoothingFactor?: number;
  gridSpacing?: number;
  gridColor?: string;
  highlightColor?: string;
}

const DEFAULT_CONFIG: Required<RendererConfig> = {
  strokeColor: STROKE_COLOR,
  baseStrokeWidth: BASE_STROKE_WIDTH,
  minStrokeWidth: MIN_STROKE_WIDTH,
  maxStrokeWidth: MAX_STROKE_WIDTH,
  smoothingFactor: STROKE_SMOOTHING_FACTOR,
  gridSpacing: GRID_SPACING,
  gridColor: GRID_COLOR,
  highlightColor: HIGHLIGHT_COLOR,
};

export class CanvasRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private scaleFactor: number;
  private showGrid: boolean = true;
  private config: Required<RendererConfig>;

  constructor(canvas: HTMLCanvasElement, scaleFactor: number = 1, config: RendererConfig = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.scaleFactor = scaleFactor;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.setupCanvas();
  }

  private setupCanvas(): void {
    // Scale the context to match the high DPI canvas
    this.ctx.scale(this.scaleFactor, this.scaleFactor);

    this.ctx.strokeStyle = this.config.strokeColor;
    this.ctx.lineWidth = this.config.baseStrokeWidth;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    // Enable anti-aliasing
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';

    // Draw initial grid
    this.drawGrid();
  }

  /**
   * Calculate stroke width based on drawing speed
   */
  calculateWidth(speed: number, previousWidth: number): number {
    // Speed thresholds (pixels per millisecond)
    const minSpeed = 0.1;  // Below this = max width (drawing very slow)
    const maxSpeed = 1.0;  // Above this = min width (drawing very fast)

    // Clamp speed to range
    const clampedSpeed = Math.max(minSpeed, Math.min(maxSpeed, speed));

    // Map speed to width (inverse relationship: fast = thin, slow = thick)
    const speedRatio = (clampedSpeed - minSpeed) / (maxSpeed - minSpeed);
    const targetWidth = this.config.maxStrokeWidth - speedRatio * (this.config.maxStrokeWidth - this.config.minStrokeWidth);

    // Smooth the width change to avoid jitter
    return previousWidth + (targetWidth - previousWidth) * this.config.smoothingFactor;
  }

  /**
   * Draw the grid background
   */
  drawGrid(): void {
    if (!this.showGrid) return;

    const width = this.canvas.width / this.scaleFactor;
    const height = this.canvas.height / this.scaleFactor;

    this.ctx.save();
    this.ctx.strokeStyle = this.config.gridColor;
    this.ctx.lineWidth = 0.5;

    // Draw vertical lines
    for (let x = 0; x <= width; x += this.config.gridSpacing) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, height);
      this.ctx.stroke();
    }

    // Draw horizontal lines
    for (let y = 0; y <= height; y += this.config.gridSpacing) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(width, y);
      this.ctx.stroke();
    }

    this.ctx.restore();
  }

  /**
   * Draw a segment with variable width (tapered between two points)
   */
  drawVariableWidthSegment(p1: Point, p2: Point, color?: string): void {
    const w1 = p1.width || this.config.baseStrokeWidth;
    const w2 = p2.width || this.config.baseStrokeWidth;

    this.ctx.save();
    this.ctx.fillStyle = color || this.config.strokeColor;

    // Calculate the perpendicular direction
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const len = Math.sqrt(dx * dx + dy * dy);

    if (len < 0.1) {
      // Points too close, just draw a circle
      this.ctx.beginPath();
      this.ctx.arc(p1.x, p1.y, w1 / 2, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.restore();
      return;
    }

    // Perpendicular unit vector
    const px = -dy / len;
    const py = dx / len;

    // Draw a quadrilateral (trapezoid) connecting the two circles
    this.ctx.beginPath();
    this.ctx.moveTo(p1.x + px * w1 / 2, p1.y + py * w1 / 2);
    this.ctx.lineTo(p2.x + px * w2 / 2, p2.y + py * w2 / 2);
    this.ctx.lineTo(p2.x - px * w2 / 2, p2.y - py * w2 / 2);
    this.ctx.lineTo(p1.x - px * w1 / 2, p1.y - py * w1 / 2);
    this.ctx.closePath();
    this.ctx.fill();

    // Draw circles at each end for smooth caps
    this.ctx.beginPath();
    this.ctx.arc(p1.x, p1.y, w1 / 2, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.beginPath();
    this.ctx.arc(p2.x, p2.y, w2 / 2, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.restore();
  }

  /**
   * Draw a single point (for single-point strokes)
   */
  drawPoint(point: Point, color?: string): void {
    this.ctx.save();
    this.ctx.fillStyle = color || this.config.strokeColor;
    this.ctx.beginPath();
    this.ctx.arc(point.x, point.y, (point.width || this.config.baseStrokeWidth) / 2, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.restore();
  }

  /**
   * Draw a complete stroke
   */
  drawStroke(stroke: Stroke, color?: string): void {
    if (stroke.points.length === 0) return;

    const strokeColor = color || this.config.strokeColor;

    // Draw each segment with variable width
    for (let i = 1; i < stroke.points.length; i++) {
      this.drawVariableWidthSegment(stroke.points[i - 1], stroke.points[i], strokeColor);
    }

    // Draw a circle at the first point if it's alone
    if (stroke.points.length === 1) {
      this.drawPoint(stroke.points[0], strokeColor);
    }
  }

  /**
   * Redraw the entire canvas with all strokes
   */
  redrawCanvas(
    strokes: Stroke[],
    currentStroke: Point[],
    isDrawing: boolean,
    highlightedStrokeIds: number[] = []
  ): void {
    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw grid first (so it's behind strokes)
    this.drawGrid();

    // Redraw all strokes with variable width
    strokes.forEach(stroke => {
      const isHighlighted = highlightedStrokeIds.includes(stroke.id);
      const color = isHighlighted ? this.config.highlightColor : this.config.strokeColor;
      this.drawStroke(stroke, color);
    });

    // Redraw current stroke if drawing
    if (isDrawing && currentStroke.length > 0) {
      for (let i = 1; i < currentStroke.length; i++) {
        this.drawVariableWidthSegment(currentStroke[i - 1], currentStroke[i]);
      }

      // Draw a circle at the first point if it's alone
      if (currentStroke.length === 1) {
        this.drawPoint(currentStroke[0]);
      }
    }
  }

  /**
   * Clear the entire canvas and redraw grid
   */
  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.drawGrid();
  }

  /**
   * Set grid visibility
   */
  setShowGrid(show: boolean): void {
    this.showGrid = show;
  }

  /**
   * Get base stroke width (for initializing new strokes)
   */
  getBaseStrokeWidth(): number {
    return this.config.baseStrokeWidth;
  }
}
