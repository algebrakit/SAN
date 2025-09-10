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

  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private strokeCounter: number = 0;
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
  };

  private stopDrawing = (event: PointerEvent) => {
    if (!this.isDrawing) return;
    
    event.preventDefault();
    this.isDrawing = false;
    
    if (this.currentStroke.length > 1) {
      const newStroke: Stroke = {
        points: [...this.currentStroke],
        id: this.strokeCounter++
      };
      this.strokes = [...this.strokes, newStroke];
    }
    
    this.currentStroke = [];
  };

  @Method()
  async clearCanvas() {
    this.strokes = [];
    this.currentStroke = [];
    this.latexResult = '';
    this.error = '';
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
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