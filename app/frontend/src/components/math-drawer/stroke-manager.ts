export interface Point {
  x: number;
  y: number;
}

export interface Stroke {
  points: Point[];
  id: number;
}

export class StrokeManager {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private strokes: Stroke[] = [];
  private currentStroke: Point[] = [];
  private isDrawing: boolean = false;
  private strokeCounter: number = 0;
  private undoStack: Stroke[][] = [];
  private redoStack: Stroke[][] = [];
  private highlightedStrokeIds: number[] = [];
  private scaleFactor: number;
  private showGrid: boolean = true;
  private gridSpacing: number = 20; // Grid spacing in pixels
  private gridColor: string = '#d8dde3'; // Subtle gray to match design
  private eraserRadius: number = 15; // Eraser radius in pixels
  private strokesToErase: Set<number> = new Set(); // Track strokes to erase during current drag

  constructor(canvas: HTMLCanvasElement, scaleFactor: number = 1) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.scaleFactor = scaleFactor;
    this.setupCanvas();
  }

  private setupCanvas() {
    // Scale the context to match the high DPI canvas
    this.ctx.scale(this.scaleFactor, this.scaleFactor);

    this.ctx.strokeStyle = '#000';
    this.ctx.lineWidth = 2.5;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    // Enable anti-aliasing
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';

    // Draw initial grid
    this.drawGrid();
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

    const point = this.getPointerPosition(event);
    this.currentStroke.push(point);

    this.ctx.beginPath();
    this.ctx.moveTo(point.x, point.y);
  }

  draw(event: PointerEvent): number[] {
    if (!this.isDrawing) return [];

    event.preventDefault();
    const point = this.getPointerPosition(event);
    this.currentStroke.push(point);

    this.ctx.lineTo(point.x, point.y);
    this.ctx.stroke();

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

  stopDrawing(event: PointerEvent): void {
    if (!this.isDrawing) return;

    event.preventDefault();
    this.isDrawing = false;

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

        // Clear highlighting and redraw canvas without the deleted strokes
        this.highlightedStrokeIds = [];
        this.redrawCanvas();
      } else {
        // Normal drawing stroke - add it
        // Save current state to undo stack before adding new stroke
        this.undoStack.push([...this.strokes]);
        this.redoStack = []; // Clear redo stack when new action is performed

        const newStroke: Stroke = {
          points: [...this.currentStroke],
          id: this.strokeCounter++
        };
        this.strokes = [...this.strokes, newStroke];

        // Clear highlighting
        this.highlightedStrokeIds = [];
      }
    }

    this.currentStroke = [];
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
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.drawGrid();
  }

  undo(): void {
    if (this.undoStack.length === 0) return;

    // Save current state to redo stack
    this.redoStack.push([...this.strokes]);

    // Restore previous state from undo stack
    this.strokes = this.undoStack.pop();

    // Redraw canvas
    this.redrawCanvas();
  }

  redo(): void {
    if (this.redoStack.length === 0) return;

    // Save current state to undo stack
    this.undoStack.push([...this.strokes]);

    // Restore state from redo stack
    this.strokes = this.redoStack.pop();

    // Redraw canvas
    this.redrawCanvas();
  }

  private drawGrid(): void {
    if (!this.showGrid) return;

    const width = this.canvas.width / this.scaleFactor;
    const height = this.canvas.height / this.scaleFactor;

    this.ctx.save();
    this.ctx.strokeStyle = this.gridColor;
    this.ctx.lineWidth = 0.5;

    // Draw vertical lines
    for (let x = 0; x <= width; x += this.gridSpacing) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, height);
      this.ctx.stroke();
    }

    // Draw horizontal lines
    for (let y = 0; y <= height; y += this.gridSpacing) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(width, y);
      this.ctx.stroke();
    }

    this.ctx.restore();
  }

  private redrawCanvas(): void {
    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw grid first (so it's behind strokes)
    this.drawGrid();

    // Redraw all strokes with appropriate colors
    this.strokes.forEach(stroke => {
      if (stroke.points.length > 0) {
        // Set color based on whether stroke is highlighted
        const isHighlighted = this.highlightedStrokeIds.includes(stroke.id);
        this.ctx.strokeStyle = isHighlighted ? '#ff0000' : '#000000';

        this.ctx.beginPath();
        this.ctx.moveTo(stroke.points[0].x, stroke.points[0].y);

        for (let i = 1; i < stroke.points.length; i++) {
          this.ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
        }

        this.ctx.stroke();
      }
    });

    // Reset stroke style to black
    this.ctx.strokeStyle = '#000000';

    // Redraw current stroke if drawing
    if (this.isDrawing && this.currentStroke.length > 0) {
      this.ctx.beginPath();
      this.ctx.moveTo(this.currentStroke[0].x, this.currentStroke[0].y);

      for (let i = 1; i < this.currentStroke.length; i++) {
        this.ctx.lineTo(this.currentStroke[i].x, this.currentStroke[i].y);
      }

      this.ctx.stroke();
    }
  }

  // Check if two line segments intersect
  private lineSegmentsIntersect(p1: Point, p2: Point, p3: Point, p4: Point): boolean {
    const ccw = (A: Point, B: Point, C: Point): boolean => {
      return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
    };

    return ccw(p1, p3, p4) !== ccw(p2, p3, p4) && ccw(p1, p2, p3) !== ccw(p1, p2, p4);
  }

  // Count how many times a stroke intersects with another stroke
  private countIntersections(stroke1: Point[], stroke2: Point[]): number {
    let count = 0;

    for (let i = 0; i < stroke1.length - 1; i++) {
      for (let j = 0; j < stroke2.length - 1; j++) {
        if (this.lineSegmentsIntersect(
          stroke1[i], stroke1[i + 1],
          stroke2[j], stroke2[j + 1]
        )) {
          count++;
        }
      }
    }

    return count;
  }

  // Detect which strokes should be deleted based on scratch gesture
  private detectScratchedStrokes(scratchStroke: Point[]): number[] {
    const scratchedIds: number[] = [];
    const intersectionThreshold = 4; // Minimum intersections to consider a stroke "scratched"

    this.strokes.forEach(stroke => {
      const intersections = this.countIntersections(scratchStroke, stroke.points);
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

  setShowGrid(show: boolean): void {
    this.showGrid = show;
    this.redrawCanvas();
  }

  // Calculate distance from a point to a line segment
  private pointToSegmentDistance(point: Point, segStart: Point, segEnd: Point): number {
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

    let xx, yy;

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

  // Check if a point is within eraser radius of any part of a stroke
  private isStrokeInEraserRadius(stroke: Stroke, point: Point, radius: number): boolean {
    // Check distance from point to each segment in the stroke
    for (let i = 0; i < stroke.points.length - 1; i++) {
      const distance = this.pointToSegmentDistance(point, stroke.points[i], stroke.points[i + 1]);
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

  // Detect strokes within eraser radius
  detectErasableStrokes(point: Point): number[] {
    const erasableIds: number[] = [];

    this.strokes.forEach(stroke => {
      if (this.isStrokeInEraserRadius(stroke, point, this.eraserRadius)) {
        erasableIds.push(stroke.id);
      }
    });

    return erasableIds;
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
