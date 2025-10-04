import { Component, h, State, Element, Method } from '@stencil/core';

// Global window interface extension for MathJax
declare global {
  interface Window {
    renderMathJax?: (element?: HTMLElement) => void;
    MathJax?: {
      typesetPromise: (elements?: HTMLElement[]) => Promise<void>;
      startup?: {
        promise?: Promise<void>;
      };
    };
  }
}

export interface Point {
  x: number;
  y: number;
}

export interface Stroke {
  points: Point[];
  id: number;
}

@Component({
  tag: 'math-drawer',
  styleUrl: 'math-drawer.css',
  shadow: false,
})
export class MathDrawer {
  @Element() el: HTMLElement;
  @State() strokes: Stroke[] = [];
  @State() currentStroke: Point[] = [];
  @State() isDrawing: boolean = false;
  @State() latexResult: string = '';
  @State() isProcessing: boolean = false;
  @State() error: string = '';
  @State() highlightedStrokeIds: number[] = [];

  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private strokeCounter: number = 0;
  private undoStack: Stroke[][] = [];
  private redoStack: Stroke[][] = [];
  // Use relative URL in production to avoid CORS
  private apiUrl: string = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5001';

  componentDidLoad() {
    this.canvas = this.el.querySelector('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.setupCanvas();
    this.setupTouchEvents();
  }

  private setupCanvas() {
    this.ctx.strokeStyle = '#000';
    this.ctx.lineWidth = 2;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
  }

  private setupTouchEvents() {
    // Prevent context menu on long press
    this.canvas.addEventListener('contextmenu', (e) => {
      e.preventDefault();
    });

    // Prevent scrolling when touching the canvas
    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        e.preventDefault();
      }
    }, { passive: false });

    this.canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1) {
        e.preventDefault();
      }
    }, { passive: false });

    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
    }, { passive: false });

    // Prevent multi-touch gestures
    this.canvas.addEventListener('gesturestart', (e) => {
      e.preventDefault();
    }, { passive: false });

    this.canvas.addEventListener('gesturechange', (e) => {
      e.preventDefault();
    }, { passive: false });

    this.canvas.addEventListener('gestureend', (e) => {
      e.preventDefault();
    }, { passive: false });
  }

  private getPointerPosition(event: PointerEvent): Point {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
  }

  private startDrawing = (event: PointerEvent) => {
    event.preventDefault();
    this.isDrawing = true;
    this.currentStroke = [];
    
    const point = this.getPointerPosition(event);
    this.currentStroke.push(point);
    
    this.ctx.beginPath();
    this.ctx.moveTo(point.x, point.y);
  };

  private draw = (event: PointerEvent) => {
    if (!this.isDrawing) return;

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
    } else if (scratchedIds.length === 0 && this.highlightedStrokeIds.length > 0) {
      this.highlightedStrokeIds = [];
      this.redrawCanvas();
    }
  };

  private stopDrawing = (event: PointerEvent) => {
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
  };

  @Method()
  async clearCanvas() {
    this.strokes = [];
    this.currentStroke = [];
    this.latexResult = '';
    this.error = '';
    this.undoStack = [];
    this.redoStack = [];
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  @Method()
  async undo() {
    if (this.undoStack.length === 0) return;

    // Save current state to redo stack
    this.redoStack.push([...this.strokes]);

    // Restore previous state from undo stack
    this.strokes = this.undoStack.pop();

    // Redraw canvas
    this.redrawCanvas();
  }

  @Method()
  async redo() {
    if (this.redoStack.length === 0) return;

    // Save current state to undo stack
    this.undoStack.push([...this.strokes]);

    // Restore state from redo stack
    this.strokes = this.redoStack.pop();

    // Redraw canvas
    this.redrawCanvas();
  }

  private redrawCanvas() {
    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

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
    const intersectionThreshold = 2; // Minimum intersections to consider a stroke "scratched"

    this.strokes.forEach(stroke => {
      const intersections = this.countIntersections(scratchStroke, stroke.points);
      if (intersections >= intersectionThreshold) {
        scratchedIds.push(stroke.id);
      }
    });

    return scratchedIds;
  }

  @Method()
  async convertToLatex() {
    if (this.strokes.length === 0) {
      this.error = 'Please draw something first';
      return;
    }

    this.isProcessing = true;
    this.error = '';
    
    try {
      // Convert strokes to the format expected by the API
      const strokeData = this.strokes.map(stroke => 
        stroke.points.map(point => [point.x, point.y])
      );
      console.log('Sending stroke data to URL:', `${this.apiUrl}/convert`);
      const response = await fetch(`${this.apiUrl}/convert`, {
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

      const result = await response.json();
      this.latexResult = result.latex;

      // Trigger MathJax rendering after LaTeX content is updated
      setTimeout(() => {
        if (window.MathJax && window.MathJax.typesetPromise) {
          const latexElement = this.el.querySelector('.latex-rendered') as HTMLElement;
          if (latexElement) {
            window.MathJax.typesetPromise([latexElement]).catch((err) => {
              console.warn('MathJax rendering failed:', err);
            });
          }
        }
      }, 100);

    } catch (err) {
      this.error = `Error: ${err.message}`;
      console.error('Conversion error:', err);
    } finally {
      this.isProcessing = false;
    }
  }

  render() {
    return (
      <div class="math-drawer-container">
        <div class="canvas-container">
          <canvas
            width={800}
            height={400}
            onPointerDown={this.startDrawing}
            onPointerMove={this.draw}
            onPointerUp={this.stopDrawing}
            onPointerOut={this.stopDrawing}
            style={{ touchAction: 'none' }}
          />
        </div>
        
        <div class="controls">
          <button onClick={() => this.undo()} disabled={this.isProcessing || this.undoStack.length === 0}>
            Undo
          </button>
          <button onClick={() => this.redo()} disabled={this.isProcessing || this.redoStack.length === 0}>
            Redo
          </button>
          <button onClick={() => this.clearCanvas()} disabled={this.isProcessing}>
            Clear
          </button>
          <button onClick={() => this.convertToLatex()} disabled={this.isProcessing || this.strokes.length === 0}>
            {this.isProcessing ? 'Converting...' : 'Convert to LaTeX'}
          </button>
        </div>
        
        {this.error && (
          <div class="error-message">
            {this.error}
          </div>
        )}
        
        {this.latexResult && (
          <div class="latex-result">
            <h3>LaTeX Result:</h3>
            <div class="latex-output">
              <code>{this.latexResult}</code>
            </div>
            <div class="latex-rendered" innerHTML={`$$${this.latexResult}$$`}></div>
          </div>
        )}
        
        <div class="info">
          <p>Draw mathematical expressions using your mouse, stylus, or finger.</p>
          <p>Strokes: {this.strokes.length}</p>
        </div>
      </div>
    );
  }
}