import { Point, Stroke } from './types';
import {
  countIntersections,
  getStrokeBoundingBox,
  getStrokesBoundingBox,
  getYPercentile
} from './geometry';
import { CanvasRenderer } from './canvas-renderer';
import { detectErasableStrokes, getDefaultEraserRadius } from './eraser';

export { Point, Stroke };

export class StrokeManager {
  private canvas: HTMLCanvasElement;
  private renderer: CanvasRenderer;
  private strokes: Stroke[] = [];
  private currentStroke: Point[] = [];
  private isDrawing: boolean = false;
  private strokeCounter: number = 0;
  private undoStack: Stroke[][] = [];
  private redoStack: Stroke[][] = [];
  private highlightedStrokeIds: number[] = [];
  private eraserRadius: number = getDefaultEraserRadius();
  private strokesToErase: Set<number> = new Set(); // Track strokes to erase during current drag
  private autoScrollReferenceStroke: Stroke | null = null; // Track stroke for auto-scroll decisions

  constructor(canvas: HTMLCanvasElement, scaleFactor: number = 1) {
    this.canvas = canvas;
    this.renderer = new CanvasRenderer(canvas, scaleFactor);
  }

  private getPointerPosition(event: PointerEvent): Point {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
  }

  startDrawing(event: PointerEvent): void {
    event.preventDefault();
    this.isDrawing = true;
    this.currentStroke = [];

    const pos = this.getPointerPosition(event);
    const point: Point = {
      x: pos.x,
      y: pos.y,
      timestamp: performance.now(),
      width: this.renderer.getBaseStrokeWidth()  // Start with base width
    };
    this.currentStroke.push(point);
  }

  draw(event: PointerEvent): number[] {
    if (!this.isDrawing) return [];

    event.preventDefault();
    const pos = this.getPointerPosition(event);
    const now = performance.now();

    // Get the previous point
    const prevPoint = this.currentStroke[this.currentStroke.length - 1];

    // Calculate speed (pixels per millisecond)
    const dx = pos.x - prevPoint.x;
    const dy = pos.y - prevPoint.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const timeDelta = now - (prevPoint.timestamp || now);
    const speed = timeDelta > 0 ? distance / timeDelta : 0;

    // Calculate width based on speed
    const width = this.renderer.calculateWidth(speed, prevPoint.width || this.renderer.getBaseStrokeWidth());

    const point: Point = {
      x: pos.x,
      y: pos.y,
      timestamp: now,
      width: width
    };
    this.currentStroke.push(point);

    // Draw the segment with variable width
    this.renderer.drawVariableWidthSegment(prevPoint, point);

    // Check for intersections in real-time and highlight strokes
    const scratchedIds = this.detectScratchedStrokes(this.currentStroke);
    if (scratchedIds.length > 0 && JSON.stringify(scratchedIds) !== JSON.stringify(this.highlightedStrokeIds)) {
      this.highlightedStrokeIds = scratchedIds;
      this.redrawCanvas();
      return scratchedIds;
    } else if (scratchedIds.length === 0 && this.highlightedStrokeIds.length > 0) {
      this.highlightedStrokeIds = [];
      this.redrawCanvas();
      return [];
    }
    return this.highlightedStrokeIds;
  }

  stopDrawing(event: PointerEvent): Stroke | null {
    if (!this.isDrawing) return null;

    event.preventDefault();
    this.isDrawing = false;

    let newStroke: Stroke | null = null;

    if (this.currentStroke.length > 1) {
      // Check if this stroke is a scratch gesture
      const scratchedIds = this.detectScratchedStrokes(this.currentStroke);

      if (scratchedIds.length > 0) {
        // This is a scratch gesture - delete the scratched strokes
        // Save current state to undo stack before deletion
        this.undoStack.push([...this.strokes]);
        this.redoStack = []; // Clear redo stack when new action is performed

        // Remove scratched strokes
        this.strokes = this.strokes.filter(stroke => !scratchedIds.includes(stroke.id));

        // Clear auto-scroll reference - scratch gesture is not normal writing
        this.autoScrollReferenceStroke = null;

        // Clear highlighting and redraw canvas without the deleted strokes
        this.highlightedStrokeIds = [];
        this.redrawCanvas();
      } else {
        // Normal drawing stroke - add it
        // Save current state to undo stack before adding new stroke
        this.undoStack.push([...this.strokes]);
        this.redoStack = []; // Clear redo stack when new action is performed

        newStroke = {
          points: [...this.currentStroke],
          id: this.strokeCounter++
        };
        this.strokes = [...this.strokes, newStroke];

        // Set as auto-scroll reference - user is actively writing
        this.autoScrollReferenceStroke = newStroke;

        // Clear highlighting
        this.highlightedStrokeIds = [];
      }
    }

    this.currentStroke = [];
    return newStroke;
  }

  // Cancel drawing (on pointer out) - don't save the stroke
  cancelDrawing(): void {
    if (!this.isDrawing) return;

    this.isDrawing = false;
    this.currentStroke = [];
    this.highlightedStrokeIds = [];
    this.redrawCanvas();
  }

  clear(): void {
    this.strokes = [];
    this.currentStroke = [];
    this.undoStack = [];
    this.redoStack = [];
    this.highlightedStrokeIds = [];
    this.autoScrollReferenceStroke = null;
    this.renderer.clear();
  }

  undo(): void {
    if (this.undoStack.length === 0) return;

    // Save current state to redo stack
    this.redoStack.push([...this.strokes]);

    // Restore previous state from undo stack
    this.strokes = this.undoStack.pop();

    // Clear auto-scroll reference - user is navigating history
    this.autoScrollReferenceStroke = null;

    // Redraw canvas
    this.redrawCanvas();
  }

  redo(): void {
    if (this.redoStack.length === 0) return;

    // Save current state to undo stack
    this.undoStack.push([...this.strokes]);

    // Restore state from redo stack
    this.strokes = this.redoStack.pop();

    // Clear auto-scroll reference - user is navigating history
    this.autoScrollReferenceStroke = null;

    // Redraw canvas
    this.redrawCanvas();
  }

  private redrawCanvas(): void {
    this.renderer.redrawCanvas(
      this.strokes,
      this.currentStroke,
      this.isDrawing,
      this.highlightedStrokeIds
    );
  }

  // Detect which strokes should be deleted based on scratch gesture
  private detectScratchedStrokes(scratchStroke: Point[]): number[] {
    const scratchedIds: number[] = [];
    const intersectionThreshold = 4; // Minimum intersections to consider a stroke "scratched"

    this.strokes.forEach(stroke => {
      const intersections = countIntersections(scratchStroke, stroke.points);
      if (intersections >= intersectionThreshold) {
        scratchedIds.push(stroke.id);
      }
    });

    return scratchedIds;
  }

  // Getters for state access
  getStrokes(): Stroke[] {
    return this.strokes;
  }

  canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  getStrokeCount(): number {
    return this.strokes.length;
  }

  hasStrokes(): boolean {
    return this.strokes.length > 0;
  }

  getStrokesBoundingBox(): { minX: number; minY: number; maxX: number; maxY: number } | null {
    return getStrokesBoundingBox(this.strokes);
  }

  getLastStrokeBoundingBox(): { minX: number; minY: number; maxX: number; maxY: number } | null {
    if (this.strokes.length === 0) return null;
    const lastStroke = this.strokes[this.strokes.length - 1];
    return getStrokeBoundingBox(lastStroke);
  }

  // Auto-scroll reference stroke management
  setAutoScrollReferenceStroke(stroke: Stroke): void {
    this.autoScrollReferenceStroke = stroke;
  }

  getAutoScrollReferenceBoundingBox(): { minX: number; minY: number; maxX: number; maxY: number } | null {
    if (!this.autoScrollReferenceStroke) return null;
    return getStrokeBoundingBox(this.autoScrollReferenceStroke);
  }

  clearAutoScrollReferenceStroke(): void {
    this.autoScrollReferenceStroke = null;
  }

  getYPercentile(percentile: number): number | null {
    return getYPercentile(this.strokes, percentile);
  }

  getStrokeCounter(): number {
    return this.strokeCounter;
  }

  setStrokes(strokes: Stroke[], strokeCounter: number): void {
    this.strokes = strokes.map(s => ({
      points: [...s.points],
      id: s.id
    }));
    this.strokeCounter = strokeCounter;
    this.currentStroke = [];
    this.undoStack = [];
    this.redoStack = [];
    this.highlightedStrokeIds = [];
    this.redrawCanvas();
  }

  setShowGrid(show: boolean): void {
    this.renderer.setShowGrid(show);
    this.redrawCanvas();
  }

  // Detect strokes within eraser radius
  detectErasableStrokes(point: Point): number[] {
    return detectErasableStrokes(this.strokes, point, this.eraserRadius);
  }

  // Start erasing mode
  startErasing(point: Point): void {
    // Save current state to undo stack before any erasing
    this.undoStack.push([...this.strokes]);
    this.redoStack = []; // Clear redo stack when new action is performed
    this.strokesToErase.clear();

    // Detect and mark strokes for erasing
    const erasableIds = this.detectErasableStrokes(point);
    erasableIds.forEach(id => this.strokesToErase.add(id));

    // Highlight strokes that will be erased
    this.highlightedStrokeIds = erasableIds;
    this.redrawCanvas();
  }

  // Continue erasing (during drag)
  continueErasing(point: Point): void {
    const erasableIds = this.detectErasableStrokes(point);
    let changed = false;

    erasableIds.forEach(id => {
      if (!this.strokesToErase.has(id)) {
        this.strokesToErase.add(id);
        changed = true;
      }
    });

    if (changed) {
      this.highlightedStrokeIds = Array.from(this.strokesToErase);
      this.redrawCanvas();
    }
  }

  // Stop erasing (on pointer up) - actually delete the strokes
  stopErasing(): void {
    if (this.strokesToErase.size > 0) {
      // Remove all marked strokes
      this.strokes = this.strokes.filter(stroke => !this.strokesToErase.has(stroke.id));
      this.strokesToErase.clear();
    } else {
      // No strokes were erased, remove the undo state we added
      this.undoStack.pop();
    }

    // Clear auto-scroll reference - user is erasing, not writing
    this.autoScrollReferenceStroke = null;

    // Clear highlighting
    this.highlightedStrokeIds = [];
    this.redrawCanvas();
  }

  // Cancel erasing (on pointer out) - don't delete, just clear highlights
  cancelErasing(): void {
    // Remove the undo state we added since no deletion happened
    if (this.strokesToErase.size > 0 || this.highlightedStrokeIds.length > 0) {
      this.undoStack.pop();
    }

    // Clear all erasing state
    this.strokesToErase.clear();
    this.highlightedStrokeIds = [];
    this.redrawCanvas();
  }
}
