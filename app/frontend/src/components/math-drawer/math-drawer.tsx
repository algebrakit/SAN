import { Component, h, State, Element, Method } from '@stencil/core';
import { StrokeManager } from './stroke-manager';

const CANVAS_WIDTH = 480;
const CANVAS_HEIGHT = 280;

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

@Component({
  tag: 'math-drawer',
  styleUrl: 'math-drawer.css',
  shadow: false,
})
export class MathDrawer {
  @Element() el: HTMLElement;
  @State() strokeCount: number = 0;
  @State() latexResult: string = '';
  @State() isProcessing: boolean = false;
  @State() error: string = '';

  private canvas: HTMLCanvasElement;
  private strokeManager: StrokeManager;
  // Use relative URL in production to avoid CORS
  private apiUrl: string = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5001';

  componentDidLoad() {
    this.canvas = this.el.querySelector('canvas');
    this.strokeManager = new StrokeManager(this.canvas);
    this.setupTouchEvents();
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

  private startDrawing = (event: PointerEvent) => {
    this.strokeManager.startDrawing(event);
  };

  private draw = (event: PointerEvent) => {
    this.strokeManager.draw(event);
  };

  private stopDrawing = (event: PointerEvent) => {
    this.strokeManager.stopDrawing(event);
    this.strokeCount = this.strokeManager.getStrokeCount();
  };

  @Method()
  async clearCanvas() {
    this.strokeManager.clear();
    this.strokeCount = 0;
    this.latexResult = '';
    this.error = '';
  }

  @Method()
  async undo() {
    this.strokeManager.undo();
    this.strokeCount = this.strokeManager.getStrokeCount();
  }

  @Method()
  async redo() {
    this.strokeManager.redo();
    this.strokeCount = this.strokeManager.getStrokeCount();
  }

  @Method()
  async convertToLatex() {
    const strokes = this.strokeManager.getStrokes();
    if (strokes.length === 0) {
      this.error = 'Please draw something first';
      return;
    }

    this.isProcessing = true;
    this.error = '';

    try {
      // Convert strokes to the format expected by the API
      const strokeData = strokes.map(stroke =>
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
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
            onPointerDown={this.startDrawing}
            onPointerMove={this.draw}
            onPointerUp={this.stopDrawing}
            onPointerOut={this.stopDrawing}
            style={{ touchAction: 'none' }}
          />
        </div>
        
        <div class="controls">
          <button onClick={() => this.undo()} disabled={this.isProcessing || !this.strokeManager?.canUndo()}>
            Undo
          </button>
          <button onClick={() => this.redo()} disabled={this.isProcessing || !this.strokeManager?.canRedo()}>
            Redo
          </button>
          <button onClick={() => this.clearCanvas()} disabled={this.isProcessing}>
            Clear
          </button>
          <button onClick={() => this.convertToLatex()} disabled={this.isProcessing || this.strokeCount === 0}>
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
          <p>Strokes: {this.strokeCount}</p>
        </div>
      </div>
    );
  }
}