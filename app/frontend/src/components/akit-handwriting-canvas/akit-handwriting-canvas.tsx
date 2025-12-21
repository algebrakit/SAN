import { Component, h, State, Element, Method, Event, EventEmitter, Prop } from '@stencil/core';
import { StrokeManager } from './stroke-manager';
import { HandwritingCanvasState } from './types';
import { UndoIcon, RedoIcon, TrashIcon, EraserIcon, SpinnerIcon, SubmitIcon, UpArrowIcon } from './icons';
import { convertStrokes } from './api-service';
import { SymbolAdjustment } from '../akit-config-handwriting/types';
import {
  CANVAS_WIDTH,
  CANVAS_HEIGHT,
  SCALE_FACTOR,
  PREPEND_AREA_WIDTH,
  BUTTON_GAP_X_LARGE,
  BUTTON_GAP_X_MIN,
  BUTTON_GAP_X_MAX,
  BUTTON_GAP_Y_LARGE,
  BUTTON_GAP_Y_MIN,
  PAN_COOLDOWN_MS,
  SCROLL_END_DELAY_MS,
  MIN_VISIBLE_STROKE_MARGIN,
  AUTO_SCROLL_MIN_GAP,
  AUTO_SCROLL_PREFERRED_GAP,
  AUTO_SCROLL_DEBOUNCE_MS,
  SYMBOL_ADJUSTMENTS
} from './config';

@Component({
  tag: 'akit-handwriting-canvas',
  styleUrl: 'akit-handwriting-canvas.scss',
  shadow: false,
})
export class AkitHandwritingCanvas {
  @Element() el: HTMLElement;
  @Prop() showSubmitButton: boolean = false;
  @Prop() symbolAdjustments: SymbolAdjustment[] = [];
  @State() strokeCount: number = 0;
  @State() latexResult: string = '';
  @State() isProcessing: boolean = false;
  @State() error: string = '';
  @State() previewState: 'current' | 'outdated' | 'updating' | 'none' = 'none';
  @State() panOffsetX: number = 0;
  @State() isEraserMode: boolean = false;
  @State() isInAutoScrollZone: boolean = false;

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
  private autoScrollTimeout: ReturnType<typeof setTimeout> | null = null;
  private _symbolAdjustments: SymbolAdjustment[];

  componentDidLoad() {
    this.canvas = this.el.querySelector('canvas');
    this.canvasContainer = this.el.querySelector('.canvas-container');
    this.setupHighDPICanvas();
    this.strokeManager = new StrokeManager(this.canvas, SCALE_FACTOR);
    this.setupTouchEvents();
    this.setupPanEvents();
  }

  componentWillUpdate() {
    this._symbolAdjustments = [...this.symbolAdjustments];
    SYMBOL_ADJUSTMENTS.forEach(adj => {
      if (!this._symbolAdjustments.find(a => a.symbol === adj.symbol)) {
        this._symbolAdjustments.push(adj);
      }
    });
  }

  private setupHighDPICanvas() {
    // Total canvas width includes prepend area on the left
    const totalWidth = PREPEND_AREA_WIDTH + CANVAS_WIDTH;

    // Set the actual size in memory (scaled up for high DPI)
    this.canvas.width = totalWidth * SCALE_FACTOR;
    this.canvas.height = CANVAS_HEIGHT * SCALE_FACTOR;

    // Set the display size (CSS pixels) - fixed size, container will clip
    this.canvas.style.width = `${totalWidth}px`;
    this.canvas.style.height = `${CANVAS_HEIGHT}px`;

    // Start scrolled to the normal writing area (after prepend area)
    if (this.canvasContainer) {
      this.panOffsetX = PREPEND_AREA_WIDTH;
      setTimeout(() => {
        if (this.canvasContainer) {
          this.canvasContainer.scrollLeft = PREPEND_AREA_WIDTH;
        }
      }, 0);
    }
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

        // Cancel pending auto-scroll since user is manually scrolling
        if (this.autoScrollTimeout) {
          clearTimeout(this.autoScrollTimeout);
          this.autoScrollTimeout = null;
          this.isInAutoScrollZone = false;
        }

        // Clamp scroll position to keep strokes visible
        const maxScrollForStrokes = this.getMaxScrollForStrokes();
        if (maxScrollForStrokes !== null && this.panOffsetX > maxScrollForStrokes) {
          this.canvasContainer.scrollLeft = maxScrollForStrokes;
          this.panOffsetX = maxScrollForStrokes;
        }

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

    // Cancel pending auto-scroll since user is panning
    if (this.autoScrollTimeout) {
      clearTimeout(this.autoScrollTimeout);
      this.autoScrollTimeout = null;
      this.isInAutoScrollZone = false;
    }

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

    // Calculate max scroll (total canvas width - container width)
    const totalWidth = PREPEND_AREA_WIDTH + CANVAS_WIDTH;
    const containerWidth = this.canvasContainer?.clientWidth || totalWidth;
    const maxScroll = Math.max(0, totalWidth - containerWidth);

    // Clamp to bounds
    let clampedOffset = Math.max(0, Math.min(maxScroll, newOffset));

    // Additionally limit scrolling left to keep strokes visible
    const maxScrollForStrokes = this.getMaxScrollForStrokes();
    if (maxScrollForStrokes !== null) {
      clampedOffset = Math.min(clampedOffset, maxScrollForStrokes);
    }

    this.panOffsetX = clampedOffset;

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

  /**
   * Calculates the maximum scroll position to keep at least part of the strokes visible.
   * When scrolling left (content moves left), panOffsetX increases.
   * We limit panOffsetX so strokes don't disappear off the left edge.
   */
  private getMaxScrollForStrokes(): number | null {
    if (!this.strokeManager) return null;

    const bbox = this.strokeManager.getStrokesBoundingBox();
    if (!bbox) return null;

    // Don't allow scrolling past the point where rightmost stroke edge
    // is only MIN_VISIBLE_STROKE_MARGIN pixels from left viewport edge
    return bbox.maxX - MIN_VISIBLE_STROKE_MARGIN;
  }

  private getCanvasPoint(event: PointerEvent): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
  }

  /**
   * Checks if the drawing position is near the right edge and auto-scrolls if needed.
   * This helps users on small screens continue writing without manual scrolling.
   */
  private checkAutoScroll(canvasX: number): void {
    if (!this.canvasContainer) return;

    // Convert canvas X to visual X (relative to visible container area)
    const visualX = canvasX - this.panOffsetX;
    const containerWidth = this.canvasContainer.clientWidth;
    const rightEdgeDistance = containerWidth - visualX;

    if (rightEdgeDistance < AUTO_SCROLL_MIN_GAP) {
      // Calculate scroll amount needed to achieve preferred gap
      // scrollAmount = preferredGap - currentGap
      const scrollAmount = AUTO_SCROLL_PREFERRED_GAP - rightEdgeDistance;

      // Calculate new scroll position, respecting max bounds
      const totalWidth = PREPEND_AREA_WIDTH + CANVAS_WIDTH;
      const maxScroll = Math.max(0, totalWidth - containerWidth);
      const newScrollLeft = Math.min(this.panOffsetX + scrollAmount, maxScroll);

      // Only scroll if we're not already at the target position
      if (newScrollLeft > this.panOffsetX) {
        this.canvasContainer.scrollTo({
          left: newScrollLeft,
          behavior: 'smooth'
        });
      }
    }
  }

  /**
   * Updates the visual indicator state for the auto-scroll trigger zone.
   * Shows the overlay when drawing position or last stroke edge is in the trigger zone.
   * @param currentX - Optional current drawing X position (canvas coordinates).
   *                   If provided, only this position is checked (used during active drawing).
   *                   If not provided, checks the last completed stroke (used after stroke ends).
   */
  private updateAutoScrollZoneState(currentX?: number): void {
    if (!this.canvasContainer) return;

    let maxX: number | null = null;

    if (currentX !== undefined) {
      // During active drawing, only check current position
      maxX = currentX;
    } else {
      // After stroke ends, check the last completed stroke
      const lastStrokeBbox = this.strokeManager?.getLastStrokeBoundingBox();
      if (lastStrokeBbox) {
        maxX = lastStrokeBbox.maxX;
      }
    }

    if (maxX === null) {
      this.isInAutoScrollZone = false;
      return;
    }

    // Check if rightmost edge is in the trigger zone
    const visualX = maxX - this.panOffsetX;
    const containerWidth = this.canvasContainer.clientWidth;
    const rightEdgeDistance = containerWidth - visualX;

    this.isInAutoScrollZone = rightEdgeDistance < AUTO_SCROLL_MIN_GAP;
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

    // Cancel pending auto-scroll since user is starting a new stroke
    if (this.autoScrollTimeout) {
      clearTimeout(this.autoScrollTimeout);
      this.autoScrollTimeout = null;
      this.isInAutoScrollZone = false;
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

      // Check if we're in the auto-scroll zone to show visual indicator
      // Pass current touch position so overlay shows during drawing
      const point = this.getCanvasPoint(event);
      this.updateAutoScrollZoneState(point.x);
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

      // Cancel any pending auto-scroll (user is still drawing)
      if (this.autoScrollTimeout) {
        clearTimeout(this.autoScrollTimeout);
      }

      // Schedule auto-scroll after debounce period
      // Use the last stroke's bounding box, not all strokes, so editing at the
      // start of a formula doesn't trigger auto-scroll to the end
      const lastStrokeBbox = this.strokeManager.getLastStrokeBoundingBox();
      if (lastStrokeBbox) {
        // Update zone state - keep overlay visible if scroll is pending
        this.updateAutoScrollZoneState();

        this.autoScrollTimeout = setTimeout(() => {
          this.checkAutoScroll(lastStrokeBbox.maxX);
          this.autoScrollTimeout = null;
          // Hide overlay after scroll completes
          this.isInAutoScrollZone = false;
        }, AUTO_SCROLL_DEBOUNCE_MS);
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

  /**
   * Clears the canvas without emitting latexChanged event. Use erase() to clear and emit.
   */
  @Method()
  async clearCanvas() {
    this.strokeManager.clear();
    this.strokeCount = 0;
    this.latexResult = '';
    this.error = '';
    this.previewState = 'none';
    this.lastConvertedStrokeCount = 0;
    this.isEraserMode = false; // Exit eraser mode when clearing
    // Cancel any pending auto-scroll
    if (this.autoScrollTimeout) {
      clearTimeout(this.autoScrollTimeout);
      this.autoScrollTimeout = null;
    }
    // Reset pan position to start of normal writing area (after prepend area)
    this.panOffsetX = PREPEND_AREA_WIDTH;
    if (this.canvasContainer) {
      this.canvasContainer.scrollLeft = PREPEND_AREA_WIDTH;
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
      const result = await convertStrokes(strokes, this._symbolAdjustments);
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

  /**
   * Calculates the button style for positioning the confirm/submit button.
   * Uses hysteresis to prevent jittery movements - only repositions when
   * the button is too close to strokes or too far away (after erasing).
   */
  private calculateButtonStyle(): { [key: string]: string } {
    const buttonStyle: { [key: string]: string } = {};
    const bbox = this.strokeManager?.getStrokesBoundingBox();

    if (!bbox) {
      return buttonStyle;
    }

    // Calculate the baseline Y position using 80th percentile
    const baselineY = this.strokeManager.getYPercentile(80) ?? bbox.maxY;

    // Hysteresis logic: only reposition if current position is too close or too far
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
      // Also reposition if button is too far from strokes (e.g., after erasing)
      if (currentGapX > BUTTON_GAP_X_MAX) {
        needsReposition = true;
      }
    }

    if (needsReposition) {
      // Position button with large gaps
      this.buttonPosX = bbox.maxX + BUTTON_GAP_X_LARGE;
      this.buttonPosY = baselineY + BUTTON_GAP_Y_LARGE;
    }

    // Adjust for scroll offset so button moves with the expression
    let visualX = this.buttonPosX - this.panOffsetX;
    let visualY = this.buttonPosY;

    // Clamp button position to keep it visible within the container
    const containerWidth = this.canvasContainer?.clientWidth || 0;
    const containerHeight = this.canvasContainer?.clientHeight || CANVAS_HEIGHT;
    const buttonSize = 48; // Approximate button size including padding
    const margin = 8; // Minimum margin from edges

    // Clamp X: keep button within visible horizontal bounds
    visualX = Math.max(margin, Math.min(visualX, containerWidth - buttonSize - margin));

    // Clamp Y: keep button within visible vertical bounds
    visualY = Math.max(margin, Math.min(visualY, containerHeight - buttonSize - margin));

    buttonStyle.left = `${visualX}px`;
    buttonStyle.top = `${visualY}px`;
    buttonStyle.right = 'auto';
    buttonStyle.bottom = 'auto';

    // Only animate left position when not scrolling (smooth reposition during writing)
    if (!this.isScrolling) {
      buttonStyle.transition = 'left 0.2s ease-out, top 0.2s ease-out, transform 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease';
    }

    return buttonStyle;
  }

  @Method()
  async getState(): Promise<HandwritingCanvasState> {
    return {
      strokes: this.strokeManager.getStrokes().map(s => ({
        points: [...s.points],
        id: s.id
      })),
      strokeCounter: this.strokeManager.getStrokeCounter(),
      scrollLeft: this.canvasContainer?.scrollLeft ?? PREPEND_AREA_WIDTH
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
          {this.isInAutoScrollZone && (
            <div class="auto-scroll-zone-overlay" style={{ width: `${AUTO_SCROLL_MIN_GAP}px` }}></div>
          )}

          {this.strokeManager?.hasStrokes() && (() => {
            const buttonStyle = this.calculateButtonStyle();
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
                    : <UpArrowIcon class="up-arrow" />
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