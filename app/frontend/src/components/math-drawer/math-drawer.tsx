import { Component, h, State, Element, Method } from '@stencil/core';
import { StrokeManager } from './stroke-manager';

const CANVAS_WIDTH = 2000;
const CANVAS_HEIGHT = 280;
const SCALE_FACTOR = 2; // 2x resolution for smoother rendering

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
    this.setupHighDPICanvas();
    this.strokeManager = new StrokeManager(this.canvas, SCALE_FACTOR);
    this.setupTouchEvents();
  }

  private setupHighDPICanvas() {
    // Set the actual size in memory (scaled up for high DPI)
    this.canvas.width = CANVAS_WIDTH * SCALE_FACTOR;
    this.canvas.height = CANVAS_HEIGHT * SCALE_FACTOR;

    // Set the display size (CSS pixels) - fixed size, container will clip
    this.canvas.style.width = `${CANVAS_WIDTH}px`;
    this.canvas.style.height = `${CANVAS_HEIGHT}px`;
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

  private renderUndoIcon() {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 7v6h6"></path>
        <path d="M21 17a9 9 0 00-9-9 9 9 0 00-6 2.3L3 13"></path>
      </svg>
    );
  }

  private renderRedoIcon() {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 7v6h-6"></path>
        <path d="M3 17a9 9 0 019-9 9 9 0 016 2.3l3 2.7"></path>
      </svg>
    );
  }

  private renderTrashIcon() {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="3 6 5 6 21 6"></polyline>
        <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"></path>
      </svg>
    );
  }

  private renderConvertIcon() {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="16 18 22 12 16 6"></polyline>
        <polyline points="8 6 2 12 8 18"></polyline>
      </svg>
    );
  }

  render() {
    return (
      <div class="math-drawer-container">
        <div class="sidebar">
          <button
            class="icon-button"
            onClick={() => this.undo()}
            disabled={this.isProcessing || !this.strokeManager?.canUndo()}
            title="Undo"
          >
            {this.renderUndoIcon()}
          </button>
          <button
            class="icon-button"
            onClick={() => this.redo()}
            disabled={this.isProcessing || !this.strokeManager?.canRedo()}
            title="Redo"
          >
            {this.renderRedoIcon()}
          </button>
          <button
            class="icon-button"
            onClick={() => this.clearCanvas()}
            disabled={this.isProcessing}
            title="Clear"
          >
            {this.renderTrashIcon()}
          </button>
          <div class="spacer"></div>
          <button
            class="icon-button"
            onClick={() => this.convertToLatex()}
            disabled={this.isProcessing || this.strokeCount === 0}
            title="Convert to LaTeX"
          >
            {this.renderConvertIcon()}
          </button>
        </div>

        <div class="canvas-wrapper">
          <div class="canvas-container">
            <canvas
              onPointerDown={this.startDrawing}
              onPointerMove={this.draw}
              onPointerUp={this.stopDrawing}
              onPointerOut={this.stopDrawing}
              style={{ touchAction: 'none' }}
            />
          </div>
          {this.latexResult && (
            <div class="inline-result">
              <div class="latex-rendered" innerHTML={`$$${this.latexResult}$$`}></div>
            </div>
          )}

          {this.error && (
            <div class="error-message">
              {this.error}
            </div>
          )}
        </div>
      </div>
    );
  }
}