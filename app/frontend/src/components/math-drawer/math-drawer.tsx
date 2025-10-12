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
  @State() previewState: 'current' | 'outdated' | 'updating' | 'none' = 'none';
  @State() panOffsetX: number = 0;
  @State() isEraserMode: boolean = false;

  private canvas: HTMLCanvasElement;
  private canvasContainer: HTMLElement;
  private strokeManager: StrokeManager;
  private lastConvertedStrokeCount: number = 0;
  // Use relative URL in production to avoid CORS
  private apiUrl: string = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5001';

  // Pan gesture tracking
  private isPanning: boolean = false;
  private lastTouchPoints: { x: number; y: number }[] = [];
  private panEndTimestamp: number = 0;
  private readonly PAN_COOLDOWN_MS: number = 200; // 0.4 second cooldown after panning

  // Eraser tracking
  private isErasing: boolean = false;

  componentDidLoad() {
    this.canvas = this.el.querySelector('canvas');
    this.canvasContainer = this.el.querySelector('.canvas-container');
    this.setupHighDPICanvas();
    this.strokeManager = new StrokeManager(this.canvas, SCALE_FACTOR);
    this.setupTouchEvents();
    this.setupPanEvents();
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

    // Prevent scrolling when touching the canvas with one finger
    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        e.preventDefault();
      } else if (e.touches.length === 2) {
        // Two-finger touch - start pan gesture
        e.preventDefault();
        this.handlePanStart(e);
      }
    }, { passive: false });

    this.canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1 && !this.isPanning) {
        e.preventDefault();
      } else if (e.touches.length >= 2) {
        // Two-finger move - pan gesture
        e.preventDefault();
        this.handlePanMove(e);
      }
    }, { passive: false });

    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      if (this.isPanning && e.touches.length < 2) {
        // Pan ends when we have less than 2 fingers
        this.handlePanEnd();
      }
    }, { passive: false });

    this.canvas.addEventListener('touchcancel', (e) => {
      e.preventDefault();
      if (this.isPanning) {
        this.handlePanEnd();
      }
    }, { passive: false });

    // Prevent multi-touch gestures (pinch zoom)
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

  private setupPanEvents() {
    // Desktop scrollbar support
    if (this.canvasContainer) {
      this.canvasContainer.addEventListener('scroll', () => {
        this.panOffsetX = this.canvasContainer.scrollLeft;
      });
    }
  }

  private handlePanStart(event: TouchEvent) {
    if (event.touches.length !== 2) return;

    this.isPanning = true;

    // Cancel any active drawing when pan starts
    this.cancelDrawing();

    // Store initial touch points
    this.lastTouchPoints = [
      { x: event.touches[0].clientX, y: event.touches[0].clientY },
      { x: event.touches[1].clientX, y: event.touches[1].clientY }
    ];
  }

  private handlePanMove(event: TouchEvent) {
    if (!this.isPanning || event.touches.length !== 2) return;

    const currentTouchPoints = [
      { x: event.touches[0].clientX, y: event.touches[0].clientY },
      { x: event.touches[1].clientX, y: event.touches[1].clientY }
    ];

    // Calculate center point movement
    const lastCenterX = (this.lastTouchPoints[0].x + this.lastTouchPoints[1].x) / 2;
    const currentCenterX = (currentTouchPoints[0].x + currentTouchPoints[1].x) / 2;

    const deltaX = currentCenterX - lastCenterX;

    // Update pan offset (invert delta for natural scrolling)
    const newOffset = this.panOffsetX - deltaX;

    // Calculate max scroll (canvas width - container width)
    const containerWidth = this.canvasContainer?.clientWidth || CANVAS_WIDTH;
    const maxScroll = Math.max(0, CANVAS_WIDTH - containerWidth);

    // Clamp to bounds
    this.panOffsetX = Math.max(0, Math.min(maxScroll, newOffset));

    // Sync with scrollbar if available
    if (this.canvasContainer) {
      this.canvasContainer.scrollLeft = this.panOffsetX;
    }

    this.lastTouchPoints = currentTouchPoints;
    // keep track of the last time we were in panning...
    this.panEndTimestamp = Date.now();
  }

  private handlePanEnd() {
    this.isPanning = false;
    this.lastTouchPoints = [];
    // Update timestamp when pan actually ends
    this.panEndTimestamp = Date.now();
  }

  private getCanvasPoint(event: PointerEvent): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
  }

  private startDrawing = (event: PointerEvent) => {
    // Don't start drawing if we're panning
    if (this.isPanning) {
      event.preventDefault();
      return;
    }

    // Don't start drawing if we just finished panning (within cooldown period)
    const timeSincePanEnd = Date.now() - this.panEndTimestamp;
    if (this.panEndTimestamp > 0 && timeSincePanEnd < this.PAN_COOLDOWN_MS) {
      event.preventDefault();
      return;
    }

    if (this.isEraserMode) {
      // Start erasing
      const point = this.getCanvasPoint(event);
      this.strokeManager.startErasing(point);
      this.isErasing = true;
    } else {
      // Normal drawing
      this.strokeManager.startDrawing(event);
    }
  };

  private draw = (event: PointerEvent) => {
    // Don't draw if we're panning
    if (this.isPanning) return;

    if (this.isEraserMode) {
      // Continue erasing
      const point = this.getCanvasPoint(event);
      this.strokeManager.continueErasing(point);
    } else {
      // Normal drawing
      this.strokeManager.draw(event);
    }
  };

  private stopDrawing = (event: PointerEvent) => {
    // Don't process stop if we're panning
    if (this.isPanning) return;

    if (this.isEraserMode && this.isErasing) {
      // Stop erasing - actually delete the strokes
      this.strokeManager.stopErasing();
      this.isErasing = false;
      this.strokeCount = this.strokeManager.getStrokeCount();

      // Update preview state if needed
      if (this.latexResult && this.strokeCount !== this.lastConvertedStrokeCount) {
        this.previewState = 'outdated';
      }
    } else if (!this.isEraserMode) {
      // Normal drawing stop
      this.strokeManager.stopDrawing(event);
      this.strokeCount = this.strokeManager.getStrokeCount();

      // Show preview in outdated state if there are strokes
      if (this.strokeCount > 0) {
        // If we have a result and stroke count changed, mark as outdated
        // If we don't have a result yet, also show as outdated (needs conversion)
        if (!this.latexResult || this.strokeCount !== this.lastConvertedStrokeCount) {
          this.previewState = 'outdated';
        }
      }
    }
  };

  private cancelDrawing = () => {
    if (this.isEraserMode && this.isErasing) {
      // Cancel erasing - don't delete, just clear highlights
      this.strokeManager.cancelErasing();
      this.isErasing = false;
    } else if (!this.isEraserMode) {
      // For normal drawing, cancel without saving
      this.strokeManager.cancelDrawing();
    }
  };

  private toggleEraserMode() {
    this.isEraserMode = !this.isEraserMode;
  }

  @Method()
  async clearCanvas() {
    this.strokeManager.clear();
    this.strokeCount = 0;
    this.latexResult = '';
    this.error = '';
    this.previewState = 'none';
    this.lastConvertedStrokeCount = 0;
    this.isEraserMode = false; // Exit eraser mode when clearing
    // Reset pan position
    this.panOffsetX = 0;
    if (this.canvasContainer) {
      this.canvasContainer.scrollLeft = 0;
    }
  }

  @Method()
  async undo() {
    this.strokeManager.undo();
    this.strokeCount = this.strokeManager.getStrokeCount();

    // Mark preview as outdated if we have a result and stroke count changed
    if (this.latexResult && this.strokeCount !== this.lastConvertedStrokeCount) {
      this.previewState = 'outdated';
    }
  }

  @Method()
  async redo() {
    this.strokeManager.redo();
    this.strokeCount = this.strokeManager.getStrokeCount();

    // Mark preview as outdated if we have a result and stroke count changed
    if (this.latexResult && this.strokeCount !== this.lastConvertedStrokeCount) {
      this.previewState = 'outdated';
    }
  }

  @Method()
  async convertToLatex() {
    const strokes = this.strokeManager.getStrokes();
    if (strokes.length === 0) {
      this.error = 'Please draw something first';
      return;
    }

    this.isProcessing = true;
    this.previewState = 'updating';
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
      this.lastConvertedStrokeCount = this.strokeCount;
      this.previewState = 'current';

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
      this.previewState = this.latexResult ? 'outdated' : 'none';
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

  private renderRefreshIcon() {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="23 4 23 10 17 10"></polyline>
        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
      </svg>
    );
  }

  private renderEraserIcon() {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 20H7L3 16L12 7L17 12M11 11L13 13"></path>
        <path d="M8.5 15.5L6 13"></path>
      </svg>
    );
  }

  private renderSpinner() {
    return (
      <svg class="spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
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
          <button
            class={`icon-button ${this.isEraserMode ? 'active' : ''}`}
            onClick={() => this.toggleEraserMode()}
            disabled={this.isProcessing}
            title="Eraser"
          >
            {this.renderEraserIcon()}
          </button>
        </div>

        <div class="canvas-wrapper">
          <div class="canvas-container">
            <canvas
              class={this.isEraserMode ? 'eraser-cursor' : ''}
              onPointerDown={this.startDrawing}
              onPointerMove={this.draw}
              onPointerUp={this.stopDrawing}
              onPointerOut={this.cancelDrawing}
              style={{ touchAction: 'none' }}
            />
          </div>
          {this.previewState !== 'none' && (
            <div
              class={`inline-result ${this.previewState}`}
              onClick={() => this.previewState === 'outdated' && this.convertToLatex()}
              title={this.previewState === 'outdated' ? 'Click to convert' : ''}
            >
              {this.latexResult ? (
                <div class="latex-rendered" innerHTML={`$$${this.latexResult}$$`}></div>
              ) : (
                <div class="empty-preview-text">Click to convert</div>
              )}
              {this.previewState === 'outdated' && (
                <div class="refresh-icon-overlay">
                  {this.renderRefreshIcon()}
                </div>
              )}
              {this.previewState === 'updating' && (
                <div class="spinner-overlay">
                  {this.renderSpinner()}
                </div>
              )}
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