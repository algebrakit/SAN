import { Component, h, State, Element, Method, Event, EventEmitter, Prop } from '@stencil/core';
import { StrokeManager } from './stroke-manager';
import { HandwritingCanvasState } from './types';
import { UndoIcon, RedoIcon, TrashIcon, EraserIcon, SpinnerIcon, CheckmarkIcon, SubmitIcon } from './icons';
import { convertStrokes } from './api-service';
import {
  CANVAS_WIDTH,
  CANVAS_HEIGHT,
  SCALE_FACTOR,
  BUTTON_GAP_X_LARGE,
  BUTTON_GAP_X_MIN,
  BUTTON_GAP_Y_LARGE,
  BUTTON_GAP_Y_MIN,
  PAN_COOLDOWN_MS,
  SCROLL_END_DELAY_MS
} from './config';

@Component({
  tag: 'akit-handwriting-canvas',
  styleUrl: 'akit-handwriting-canvas.css',
  shadow: false,
})
export class AkitHandwritingCanvas {
  @Element() el: HTMLElement;
  @Prop() showSubmitButton: boolean = false;
  @State() strokeCount: number = 0;
  @State() latexResult: string = '';
  @State() isProcessing: boolean = false;
  @State() error: string = '';
  @State() previewState: 'current' | 'outdated' | 'updating' | 'none' = 'none';
  @State() panOffsetX: number = 0;
  @State() isEraserMode: boolean = false;

  // Tracked button position for hysteresis (prevents jittery movements)
  private buttonPosX: number | null = null;
  private buttonPosY: number | null = null;

  @Event() latexChanged: EventEmitter<{ latex: string }>;
  @Event() submitted: EventEmitter<{ latex: string }>;

  private canvas: HTMLCanvasElement;
  private canvasContainer: HTMLElement;
  private strokeManager: StrokeManager;
  private lastConvertedStrokeCount: number = 0;

  // Pan gesture tracking
  private isPanning: boolean = false;
  private lastTouchPoints: { x: number; y: number }[] = [];
  private panEndTimestamp: number = 0;

  // Eraser tracking
  private isErasing: boolean = false;

  // Scroll tracking for button transition
  private isScrolling: boolean = false;
  private scrollEndTimeout: ReturnType<typeof setTimeout> | null = null;

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

        // Track scrolling state for button transition
        this.isScrolling = true;
        if (this.scrollEndTimeout) {
          clearTimeout(this.scrollEndTimeout);
        }
        this.scrollEndTimeout = setTimeout(() => {
          this.isScrolling = false;
        }, SCROLL_END_DELAY_MS);
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
    if (this.panEndTimestamp > 0 && timeSincePanEnd < PAN_COOLDOWN_MS) {
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


  async erase() {
    this.clearCanvas();
    this.latexChanged.emit({ latex: '' });
  }

  @Method()
  async clearCanvas() {
    this.strokeManager.clear();
    this.strokeCount = 0;
    this.latexResult = '';
    this.latexChanged.emit({ latex: '' });
    this.error = '';
    this.previewState = 'none';
    this.lastConvertedStrokeCount = 0;
    this.isEraserMode = false; // Exit eraser mode when clearing
    // Reset pan position
    this.panOffsetX = 0;
    if (this.canvasContainer) {
      this.canvasContainer.scrollLeft = 0;
    }
    // Reset button position for hysteresis
    this.buttonPosX = null;
    this.buttonPosY = null;
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
      const result = await convertStrokes(strokes);
      this.latexResult = result.latex;
      this.latexChanged.emit({ latex: result.latex });
      this.lastConvertedStrokeCount = this.strokeCount;
      this.previewState = 'current';
    } catch (err) {
      this.error = `Error: ${err.message}`;
      console.error('Conversion error:', err);
      this.previewState = this.latexResult ? 'outdated' : 'none';
    } finally {
      this.isProcessing = false;
    }
  }

  private handleSubmit() {
    if (this.latexResult) {
      this.submitted.emit({ latex: this.latexResult });
    }
  }

  @Method()
  async getState(): Promise<HandwritingCanvasState> {
    return {
      strokes: this.strokeManager.getStrokes().map(s => ({
        points: [...s.points],
        id: s.id
      })),
      strokeCounter: this.strokeManager.getStrokeCounter(),
      scrollLeft: this.canvasContainer?.scrollLeft ?? 0
    };
  }

  @Method()
  async restoreState(state: HandwritingCanvasState): Promise<void> {
    // Restore strokes
    this.strokeManager.setStrokes(state.strokes, state.strokeCounter);
    this.strokeCount = this.strokeManager.getStrokeCount();

    // Restore scroll position
    this.panOffsetX = state.scrollLeft;
    if (this.canvasContainer) {
      this.canvasContainer.scrollLeft = state.scrollLeft;
    }

    // Reset other state
    this.latexResult = '';
    this.error = '';
    this.previewState = this.strokeCount > 0 ? 'outdated' : 'none';
    this.lastConvertedStrokeCount = 0;
    this.isEraserMode = false;
    this.buttonPosX = null;
    this.buttonPosY = null;
  }

  render() {
    return (
      <div class="akit-handwriting-canvas-container">
        <div class="canvas-wrapper">
          <div class="toolbar">
            <button
              class="icon-button"
              onClick={() => this.undo()}
              disabled={this.isProcessing || !this.strokeManager?.canUndo()}
              title="Undo"
            >
              <UndoIcon />
            </button>
            <button
              class="icon-button"
              onClick={() => this.redo()}
              disabled={this.isProcessing || !this.strokeManager?.canRedo()}
              title="Redo"
            >
              <RedoIcon />
            </button>
            <button
              class="icon-button"
              onClick={() => this.erase()}
              disabled={this.isProcessing}
              title="Clear"
            >
              <TrashIcon />
            </button>
            <button
              class={`icon-button ${this.isEraserMode ? 'active' : ''}`}
              onClick={() => this.toggleEraserMode()}
              disabled={this.isProcessing}
              title="Eraser"
            >
              <EraserIcon />
            </button>
          </div>
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

          {this.strokeManager?.hasStrokes() && (() => {
            const bbox = this.strokeManager.getStrokesBoundingBox();
            const buttonStyle: { [key: string]: string } = {};

            if (bbox) {
              // Calculate the baseline Y position using 80th percentile
              const baselineY = this.strokeManager.getYPercentile(80) ?? bbox.maxY;

              // Hysteresis logic: only reposition if current position is too close
              // This prevents small distracting movements after each stroke
              let needsReposition = false;

              if (this.buttonPosX === null || this.buttonPosY === null) {
                // No position set yet, need to position
                needsReposition = true;
              } else {
                // Check if strokes have gotten too close to current button position
                const currentGapX = this.buttonPosX - bbox.maxX;
                const currentGapY = this.buttonPosY - baselineY;

                // Reposition if gap in X is too small OR gap in Y is too small (strokes above button)
                if (currentGapX < BUTTON_GAP_X_MIN || currentGapY < BUTTON_GAP_Y_MIN) {
                  needsReposition = true;
                }
              }

              if (needsReposition) {
                // Position button with large gaps
                this.buttonPosX = bbox.maxX + BUTTON_GAP_X_LARGE;
                this.buttonPosY = baselineY + BUTTON_GAP_Y_LARGE;
              }

              // Adjust for scroll offset so button moves with the expression
              const visualX = this.buttonPosX - this.panOffsetX;
              buttonStyle.left = `${visualX}px`;
              buttonStyle.top = `${this.buttonPosY}px`;
              buttonStyle.right = 'auto';
              buttonStyle.bottom = 'auto';

              // Only animate left position when not scrolling (smooth reposition during writing)
              if (!this.isScrolling) {
                buttonStyle.transition = 'left 0.2s ease-out, top 0.2s ease-out, transform 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease';
              }
            }

            const isSubmitMode = this.showSubmitButton && this.previewState === 'current';

            return (
              <button
                class={`confirm-button ${isSubmitMode ? 'submit-mode' : ''}`}
                style={buttonStyle}
                onClick={() => isSubmitMode ? this.handleSubmit() : this.convertToLatex()}
                disabled={this.isProcessing}
                title={isSubmitMode ? "Submit answer" : "Convert to LaTeX"}
              >
                {this.isProcessing
                  ? <SpinnerIcon class="spinner" />
                  : isSubmitMode
                    ? <SubmitIcon class="submit-icon" />
                    : <CheckmarkIcon class="checkmark" />
                }
              </button>
            );
          })()}

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