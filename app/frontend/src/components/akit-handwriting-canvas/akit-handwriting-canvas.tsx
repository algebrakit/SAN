import { Component, h, State, Element, Method, Event, EventEmitter, Prop } from '@stencil/core';
import { StrokeManager } from './stroke-manager';
import { HandwritingCanvasState, SymbolAdjustment } from './types';
import { UndoIcon, RedoIcon, TrashIcon, EraserIcon, SubmitIcon } from './icons';
import { convertStrokes } from './api-service';
import {
  CANVAS_WIDTH,
  CANVAS_HEIGHT,
  CANVAS_VISIBLE_HEIGHT,
  SCALE_FACTOR,
  PREPEND_AREA_WIDTH,
  PAN_COOLDOWN_MS,
  MIN_VISIBLE_STROKE_MARGIN,
  AUTO_SCROLL_MIN_GAP,
  AUTO_SCROLL_PREFERRED_GAP,
  AUTO_SCROLL_DEBOUNCE_MS,
  AUTO_SCROLL_DEBOUNCE_MAX_MS,
  AUTO_SCROLL_DEBOUNCE_MARGIN,
  AUTO_SCROLL_MIN_STROKE_SAMPLES,
  AUTO_CONVERT_DEBOUNCE_MS,
  SYMBOL_ADJUSTMENTS,
  VERTICAL_SCROLL_MAX
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
  @State() panOffsetX: number = 0;
  @State() panOffsetY: number = VERTICAL_SCROLL_MAX;  // Start centered vertically
  @State() isEraserMode: boolean = false;
  @State() isInAutoScrollZone: boolean = false;

  // Preview bar state
  @State() previewLatex: string = '';
  @State() isAutoConverting: boolean = false;
  @State() conversionError: string = '';
  @State() isDrawing: boolean = false;
  @State() isErasing: boolean = false;
  @State() isPreviewStale: boolean = false; // True when strokes changed but conversion not yet done

  @Event() latexChanged: EventEmitter<{ latex: string }>;
  @Event() submitted: EventEmitter<{ latex: string }>;

  private canvas: HTMLCanvasElement;
  private canvasContainer: HTMLElement;
  private strokeManager: StrokeManager;

  // Pan gesture tracking
  private isPanning: boolean = false;
  private lastTouchPoints: { x: number; y: number }[] = [];
  private panEndTimestamp: number = 0;

  // Timers
  private autoScrollTimeout: ReturnType<typeof setTimeout> | null = null;
  private autoConvertTimeout: ReturnType<typeof setTimeout> | null = null;
  private _symbolAdjustments: SymbolAdjustment[];

  // Inter-stroke timing data collection
  private lastStrokeEndTimestamp: number = 0;
  private interStrokeIntervals: number[] = [];

  /**
   * Calculates adaptive auto-scroll debounce time based on observed inter-stroke intervals.
   * Uses Q3 (75th percentile) * margin as a robust upper bound estimate.
   * Falls back to default if not enough samples are collected.
   */
  private getAdaptiveDebounceMs(): number {
    // Need minimum samples before using adaptive timing
    if (this.interStrokeIntervals.length < AUTO_SCROLL_MIN_STROKE_SAMPLES) {
      return AUTO_SCROLL_DEBOUNCE_MS;
    }

    // Calculate Q3 (75th percentile)
    const sorted = [...this.interStrokeIntervals].sort((a, b) => a - b);
    const q3Index = Math.floor(sorted.length * 0.75);
    const q3 = sorted[q3Index];

    // Apply safety margin and cap at maximum
    const adaptive = Math.min(q3 * AUTO_SCROLL_DEBOUNCE_MARGIN, AUTO_SCROLL_DEBOUNCE_MAX_MS);

    return adaptive;
  }

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

    // Set the display size (CSS pixels) - canvas is taller to allow vertical scroll
    this.canvas.style.width = `${totalWidth}px`;
    this.canvas.style.height = `${CANVAS_HEIGHT}px`;

    // Set container to visible height only (clips the extra vertical space)
    if (this.canvasContainer) {
      this.canvasContainer.style.height = `${CANVAS_VISIBLE_HEIGHT}px`;
    }

    // Start scrolled to the normal writing area (after prepend area) and centered vertically
    if (this.canvasContainer) {
      this.panOffsetX = PREPEND_AREA_WIDTH;
      this.panOffsetY = VERTICAL_SCROLL_MAX;  // Center vertically
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
    // Desktop scrollbar support (horizontal)
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
      });

      // Desktop wheel support for vertical scrolling (Shift+wheel or trackpad vertical)
      this.canvasContainer.addEventListener('wheel', (e) => {
        // Use deltaY for vertical scrolling when Shift is held, or when it's a vertical scroll
        if (e.shiftKey || Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
          const deltaY = e.deltaY;
          if (Math.abs(deltaY) > 0) {
            e.preventDefault();
            const newOffsetY = this.panOffsetY + deltaY * 0.5; // Scale down for smoother scrolling
            this.panOffsetY = Math.max(0, Math.min(VERTICAL_SCROLL_MAX * 2, newOffsetY));
          }
        }
      }, { passive: false });
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

    // Calculate center point movement (X and Y)
    const lastCenterX = (this.lastTouchPoints[0].x + this.lastTouchPoints[1].x) / 2;
    const lastCenterY = (this.lastTouchPoints[0].y + this.lastTouchPoints[1].y) / 2;
    const currentCenterX = (currentTouchPoints[0].x + currentTouchPoints[1].x) / 2;
    const currentCenterY = (currentTouchPoints[0].y + currentTouchPoints[1].y) / 2;

    const deltaX = currentCenterX - lastCenterX;
    const deltaY = currentCenterY - lastCenterY;

    // Update horizontal pan offset (invert delta for natural scrolling)
    const newOffsetX = this.panOffsetX - deltaX;

    // Calculate max horizontal scroll (total canvas width - container width)
    const totalWidth = PREPEND_AREA_WIDTH + CANVAS_WIDTH;
    const containerWidth = this.canvasContainer?.clientWidth || totalWidth;
    const maxScrollX = Math.max(0, totalWidth - containerWidth);

    // Clamp horizontal to bounds
    let clampedOffsetX = Math.max(0, Math.min(maxScrollX, newOffsetX));

    // Additionally limit scrolling left to keep strokes visible
    const maxScrollForStrokes = this.getMaxScrollForStrokes();
    if (maxScrollForStrokes !== null) {
      clampedOffsetX = Math.min(clampedOffsetX, maxScrollForStrokes);
    }

    this.panOffsetX = clampedOffsetX;

    // Update vertical pan offset (invert delta for natural scrolling)
    const newOffsetY = this.panOffsetY - deltaY;
    // Clamp vertical to bounds: 0 to VERTICAL_SCROLL_MAX * 2
    this.panOffsetY = Math.max(0, Math.min(VERTICAL_SCROLL_MAX * 2, newOffsetY));

    // Sync horizontal with scrollbar if available
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
   * Shows the overlay when the current drawing position is in the trigger zone.
   * @param currentX - Current drawing X position (canvas coordinates).
   */
  private updateAutoScrollZoneState(currentX: number): void {
    if (!this.canvasContainer) return;

    // Check if current drawing position is in the trigger zone
    const visualX = currentX - this.panOffsetX;
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

    // Cancel pending auto-convert and clear preview while drawing
    if (this.autoConvertTimeout) {
      clearTimeout(this.autoConvertTimeout);
      this.autoConvertTimeout = null;
    }

    // Track inter-stroke interval for adaptive debounce
    if (this.lastStrokeEndTimestamp > 0) {
      const interval = Date.now() - this.lastStrokeEndTimestamp;
      this.interStrokeIntervals.push(interval);
    }

    this.isDrawing = true;
    this.isPreviewStale = true; // Mark preview as stale until new conversion completes
    // Keep previewLatex visible (grayed out) while drawing
    this.conversionError = '';

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
    // Don't draw if we're panning or not actively drawing
    if (this.isPanning || !this.isDrawing) return;

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

    this.isDrawing = false;

    if (this.isEraserMode && this.isErasing) {
      // Stop erasing - actually delete the strokes
      this.strokeManager.stopErasing();
      this.isErasing = false;
      this.strokeCount = this.strokeManager.getStrokeCount();

      // Schedule auto-convert after erasing
      this.scheduleAutoConvert();
    } else if (!this.isEraserMode) {
      // Normal drawing stop
      this.strokeManager.stopDrawing(event);
      this.strokeCount = this.strokeManager.getStrokeCount();
      this.lastStrokeEndTimestamp = Date.now();  // Record for inter-stroke timing

      // Schedule auto-convert
      this.scheduleAutoConvert();

      // Cancel any pending auto-scroll (user is still drawing)
      if (this.autoScrollTimeout) {
        clearTimeout(this.autoScrollTimeout);
      }

      // Schedule auto-scroll after debounce period
      // Use the last stroke's bounding box, not all strokes, so editing at the
      // start of a formula doesn't trigger auto-scroll to the end
      const lastStrokeBbox = this.strokeManager.getLastStrokeBoundingBox();
      if (lastStrokeBbox) {
        // Check if stroke ended in trigger zone - keep overlay visible during debounce
        this.updateAutoScrollZoneState(lastStrokeBbox.maxX);

        if (this.isInAutoScrollZone) {
          const scrollX = lastStrokeBbox.maxX;
          // Schedule auto-scroll and hide overlay after it completes
          this.autoScrollTimeout = setTimeout(() => {
            this.checkAutoScroll(scrollX);
            this.isInAutoScrollZone = false;
            this.autoScrollTimeout = null;
          }, this.getAdaptiveDebounceMs());
        }
      } else {
        // No stroke - hide overlay
        this.isInAutoScrollZone = false;
      }
    }
  };

  private cancelDrawing = () => {
    this.isDrawing = false;

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
    this.previewLatex = '';
    this.conversionError = '';
    this.isPreviewStale = false;
    this.isEraserMode = false; // Exit eraser mode when clearing
    // Cancel any pending auto-scroll and auto-convert
    if (this.autoScrollTimeout) {
      clearTimeout(this.autoScrollTimeout);
      this.autoScrollTimeout = null;
    }
    if (this.autoConvertTimeout) {
      clearTimeout(this.autoConvertTimeout);
      this.autoConvertTimeout = null;
    }
    // Reset pan position to start of normal writing area (after prepend area) and center vertically
    this.panOffsetX = PREPEND_AREA_WIDTH;
    this.panOffsetY = VERTICAL_SCROLL_MAX;  // Center vertically
    if (this.canvasContainer) {
      this.canvasContainer.scrollLeft = PREPEND_AREA_WIDTH;
    }
    // Reset inter-stroke timing data for new session
    this.lastStrokeEndTimestamp = 0;
    this.interStrokeIntervals = [];
  }

  @Method()
  async undo() {
    this.strokeManager.undo();
    this.strokeCount = this.strokeManager.getStrokeCount();

    // Schedule auto-convert after undo
    this.scheduleAutoConvert();
  }

  @Method()
  async redo() {
    this.strokeManager.redo();
    this.strokeCount = this.strokeManager.getStrokeCount();

    // Schedule auto-convert after redo
    this.scheduleAutoConvert();
  }

  /**
   * Schedules auto-conversion after a debounce period.
   * Cancels any pending conversion and starts a new timer.
   */
  private scheduleAutoConvert() {
    // Cancel any pending auto-convert
    if (this.autoConvertTimeout) {
      clearTimeout(this.autoConvertTimeout);
    }

    // Only schedule if there are strokes
    if (this.strokeCount > 0) {
      this.autoConvertTimeout = setTimeout(() => {
        this.autoConvert();
        this.autoConvertTimeout = null;
      }, AUTO_CONVERT_DEBOUNCE_MS);
    } else {
      // No strokes - clear preview
      this.previewLatex = '';
      this.conversionError = '';
    }
  }

  /**
   * Performs auto-conversion of strokes to LaTeX.
   * Updates preview state but does not emit latexChanged event.
   */
  private async autoConvert() {
    const strokes = this.strokeManager.getStrokes();
    if (strokes.length === 0) {
      this.previewLatex = '';
      return;
    }

    this.isAutoConverting = true;
    this.conversionError = '';

    try {
      const result = await convertStrokes(strokes, this._symbolAdjustments);
      this.previewLatex = result.latex;
      this.isPreviewStale = false; // Preview is now up-to-date
    } catch (err) {
      this.conversionError = 'Conversion failed';
      console.error('Auto-conversion error:', err);
    } finally {
      this.isAutoConverting = false;
    }
  }

  /**
   * Handles accepting the current preview LaTeX.
   * Emits the latexChanged event and optionally the submitted event.
   */
  private handleAccept() {
    if (!this.previewLatex) return;

    this.latexChanged.emit({ latex: this.previewLatex });

    if (this.showSubmitButton) {
      this.submitted.emit({ latex: this.previewLatex });
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
    this.previewLatex = '';
    this.conversionError = '';
    this.isPreviewStale = false;
    this.isEraserMode = false;

    // Trigger auto-convert if there are strokes
    if (this.strokeCount > 0) {
      this.isPreviewStale = true; // Will be stale until conversion completes
      this.scheduleAutoConvert();
    }
  }

  render() {
    return (
      <div class="akit-handwriting-canvas-container">
        {/* Preview bar - shown when there are strokes */}
        {this.strokeCount > 0 && (
          <div class={`preview-bar ${this.isPreviewStale ? 'stale' : ''}`}>
            {this.conversionError && !this.isPreviewStale ? (
              <div class="preview-error">
                <span>{this.conversionError}</span>
              </div>
            ) : this.previewLatex ? (
              <div class="preview-content">
                <div
                  class="preview-latex">
                  <akit-latex latex={this.previewLatex}></akit-latex>
                </div>
                <button
                  class="accept-button"
                  onClick={() => this.handleAccept()}
                  disabled={this.isPreviewStale || this.isAutoConverting}
                  title="Accept"
                >
                  <SubmitIcon />
                </button>
              </div>
            ) : null}
          </div>
        )}

        <div class="canvas-wrapper">
          <div class={`toolbar ${this.isDrawing || this.isErasing ? 'drawing-active' : ''}`}>
            <button
              class="icon-button"
              onClick={() => this.undo()}
              disabled={this.isAutoConverting || !this.strokeManager?.canUndo()}
              title="Undo"
            >
              <UndoIcon />
            </button>
            <button
              class="icon-button"
              onClick={() => this.redo()}
              disabled={this.isAutoConverting || !this.strokeManager?.canRedo()}
              title="Redo"
            >
              <RedoIcon />
            </button>
            <button
              class="icon-button"
              onClick={() => this.erase()}
              disabled={this.isAutoConverting}
              title="Clear"
            >
              <TrashIcon />
            </button>
            <button
              class={`icon-button ${this.isEraserMode ? 'active' : ''}`}
              onClick={() => this.toggleEraserMode()}
              disabled={this.isAutoConverting}
              title="Eraser"
            >
              <EraserIcon />
            </button>
          </div>
          <div class="canvas-container">
            <canvas
              class={`${this.isEraserMode ? 'eraser-cursor' : ''} ${this.isDrawing || this.isErasing ? 'drawing-active' : ''}`}
              onPointerDown={this.startDrawing}
              onPointerMove={this.draw}
              onPointerUp={this.stopDrawing}
              onPointerOut={this.cancelDrawing}
              style={{ touchAction: 'none', transform: `translateY(${-this.panOffsetY}px)` }}
            />
          </div>
          {this.isInAutoScrollZone && (
            <div class="auto-scroll-zone-overlay" style={{ width: `${AUTO_SCROLL_MIN_GAP}px` }}></div>
          )}
        </div>
      </div>
    );
  }
}