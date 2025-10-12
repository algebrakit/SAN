# CLAUDE.md - Math Expression Drawer Frontend

This file provides guidance to Claude Code when working with the frontend component of the handwritten math expression recognition app.

## Repository Overview

This is a **StencilJS-based web component** that provides a drawing interface for handwritten mathematical expressions. The component integrates with the SAN (Syntax-Aware Network) backend to convert hand-drawn expressions into LaTeX format.

The frontend is a standalone, embeddable web component with a modern, compact UI featuring icon-based controls and real-time preview.

## Key Commands

### Development
```bash
# Install dependencies
npm install

# Start dev server with hot reload (runs at http://localhost:3334)
npm start

# Build for production
npm run build

# Generate new components
npm run generate
```

### Testing
```bash
# Run all tests
npm test

# Run tests in watch mode
npm test.watch
```

### Playwright Testing
The component can be tested using Playwright MCP at `localhost:33333`

## Architecture Overview

### Technology Stack
- **StencilJS 4.7**: Web component compiler for building reusable UI components
- **TypeScript 5.2**: Type-safe JavaScript
- **Canvas API**: Core drawing functionality
- **Pointer Events API**: Cross-device input handling (mouse, stylus, touch)
- **CSS Flexbox**: Responsive layout system

### Component Structure

#### Main Component: `math-drawer`
**Location**: `src/components/math-drawer/`

**Files**:
- `math-drawer.tsx` - Main component logic, state management, event handlers
- `math-drawer.css` - All styling for component UI
- `stroke-manager.ts` - Stroke drawing, grid rendering, canvas operations
- `readme.md` - Auto-generated API documentation

**Key Constants** (in `math-drawer.tsx`):
```typescript
CANVAS_WIDTH = 2000    // Internal canvas resolution
CANVAS_HEIGHT = 280    // Internal canvas height
SCALE_FACTOR = 2       // High-DPI scaling
```

### Core Features

#### 1. Drawing System
- **Canvas-based drawing** with high-DPI support (2x scaling)
- **Multi-input support**: Mouse, stylus (with pressure), touch
- **Stroke management**: Add, remove, undo/redo with history
- **Grid background**: Optional grid overlay for alignment
- **Panning**: Horizontal scrolling for wide expressions (2000px canvas)

#### 2. Tool System
**Sidebar Icons** (vertical layout on left):
- **Undo** (↶): Revert last stroke
- **Redo** (↷): Restore undone stroke
- **Clear** (🗑️): Clear entire canvas
- **Grid** (⊞): Toggle grid visibility
- **Eraser** (⌫): Radius-based stroke eraser (15px radius)

#### 3. LaTeX Conversion
- **Auto-update preview**: Shows conversion status with color-coded states
  - Green border: Current/up-to-date
  - Amber border: Outdated (click to update)
  - Loading state: Converting...
- **Inline result display**: Bottom-right corner overlay
- **Click-to-convert**: Preview is the sole conversion trigger
- **API integration**: POST to `/convert` endpoint

#### 4. Eraser Tool
- **Toggle mode**: Click eraser icon to activate/deactivate
- **Radius detection**: 15px detection radius, 10px visual indicator
- **Visual feedback**: Red circular cursor, highlights strokes to delete
- **Grouped undo**: All deletions in one drag = single undo action
- **Custom cursor**: SVG-based red circle cursor

#### 5. Canvas Panning
- **Desktop**: Horizontal scrollbar (canvas 2000px, container ~845px)
- **Mobile**: Two-finger gesture for panning (prevents drawing)
- **Smart positioning**: Inline result stays in viewport during pan
- **Grid alignment**: Grid stays aligned during pan operations
- **Bounds checking**: Cannot pan beyond canvas edges

### State Management

**Component State Variables**:
```typescript
strokes: Array<Array<[number, number]>>  // All current strokes
undoneStrokes: Array<...>                // Undo history
latexResult: string                      // LaTeX output
previewState: 'none' | 'current' | 'outdated' | 'updating'
lastConvertedStrokeCount: number        // Track when conversion happened
isEraserMode: boolean                   // Eraser tool active
showGrid: boolean                       // Grid visibility
panOffsetX: number                      // Horizontal pan position
```

### API Integration

**Endpoint**: `http://localhost:5001/convert`

**Request Format**:
```typescript
POST /convert
{
  "strokes": [
    [[x1, y1], [x2, y2], ...],  // Each stroke as array of points
    [[x3, y3], [x4, y4], ...],
    ...
  ],
  "stroke_length": 25  // Normalization parameter
}
```

**Response**:
```typescript
{
  "latex": "x^2 + y = 5"  // LaTeX string
}
```

### Project Structure

```
app/frontend/
├── src/
│   ├── components/
│   │   └── math-drawer/
│   │       ├── math-drawer.tsx      # Main component (state, events, UI)
│   │       ├── math-drawer.css      # All styles
│   │       ├── stroke-manager.ts    # Canvas operations, grid, drawing
│   │       └── readme.md            # Auto-gen API docs
│   ├── index.html                   # Host page
│   ├── index.ts                     # Entry point
│   └── components.d.ts              # Type definitions
├── dist/                            # Build output
├── www/                             # Dev server output
├── package.json
├── stencil.config.ts               # Stencil configuration
├── tsconfig.json                   # TypeScript config
└── CLAUDE.md                       # This file
```

## Important Implementation Details

### Coordinate System
- Canvas uses **internal resolution** (2000×280) with 2x scale factor
- All pointer events are converted to canvas coordinates
- Pan offset is tracked separately and applied during rendering
- Grid rendering accounts for pan offset

### Drawing Flow
1. `pointerdown` → Start new stroke, check if eraser mode
2. `pointermove` → Add points to current stroke OR erase if in eraser mode
3. `pointerup` → Finalize stroke, update preview state
4. Two-finger touch → Pan mode (no drawing)

### Eraser Implementation
- Distance-based detection using point-to-segment distance calculation
- Real-time highlighting of strokes to be erased (red)
- Deletions grouped per drag for single undo action
- Custom SVG cursor with red circle overlay

### Preview States
- `none` → No preview shown (initial state)
- `outdated` → Show "Click to convert" or refresh icon (amber border)
- `updating` → Show loading state during API call
- `current` → Show result with green border

### Grid System
- Rendered on canvas using `stroke-manager.ts`
- 40px spacing, 1px lines, light gray color
- Toggle via grid icon in sidebar
- Redrawn on pan/zoom operations

### Responsive Behavior
- Canvas container uses flexbox with `overflow-x: auto`
- Scrollbar visible on desktop, hidden on touch devices
- Inline result positioned relative to viewport, not canvas
- Touch events prevent default to avoid browser zoom/scroll

## Code Style & Patterns

### StencilJS Conventions
- Use `@Component` decorator for component definition
- Use `@State()` for reactive state
- Use `@Method()` for public API methods
- Use `@Element()` to access host element

### Event Handling Pattern
```typescript
// Pointer events bound in render()
onPointerDown={(e) => this.startDrawing(e)}
onPointerMove={(e) => this.draw(e)}
onPointerUp={() => this.stopDrawing()}
```

### Stroke Management Pattern
```typescript
// Always use spread operator for immutability
this.strokes = [...this.strokes, newStroke];
this.undoneStrokes = [];  // Clear redo stack
```

### Canvas Rendering Pattern
```typescript
// Clear and redraw everything on each update
const ctx = canvas.getContext('2d');
ctx.clearRect(0, 0, canvas.width, canvas.height);
if (this.showGrid) drawGrid(ctx, this.panOffsetX);
drawAllStrokes(ctx, this.strokes, this.panOffsetX);
```

## Testing Strategy

### Manual Testing Checklist
- [ ] Drawing with mouse works correctly
- [ ] Drawing with stylus captures pressure
- [ ] Touch drawing works on mobile
- [ ] Undo/redo maintain correct state
- [ ] Clear button resets everything
- [ ] Grid toggle shows/hides grid
- [ ] Eraser deletes strokes correctly
- [ ] Eraser undo groups deletions
- [ ] Preview shows correct states
- [ ] Click preview to convert works
- [ ] Panning with scrollbar (desktop)
- [ ] Two-finger pan (mobile - requires device)
- [ ] Inline result stays in viewport
- [ ] API conversion returns LaTeX

### Known Testing Gaps
- Two-finger pan gesture requires physical mobile device testing
- Stylus pressure sensitivity varies by device
- Browser zoom interaction with two-finger pan

## Troubleshooting

### Common Issues

**Canvas not responding to input**
- Check `touch-action: none` in CSS
- Verify pointer event handlers are bound
- Check z-index of canvas vs other elements

**Coordinates offset after panning**
- Ensure `panOffsetX` is subtracted when converting pointer → canvas coords
- Ensure `panOffsetX` is added when rendering strokes
- Check scroll position is synced with panOffsetX

**Preview not updating**
- Verify `previewState` transitions: none → outdated → updating → current
- Check `lastConvertedStrokeCount` is updated after conversion
- Ensure stroke count changes trigger outdated state

**Eraser not working**
- Check `isEraserMode` state is true
- Verify distance calculation in point-to-segment logic
- Check `eraserRadius` constant (should be 15)

**Grid misaligned**
- Ensure grid rendering uses `panOffsetX`
- Check grid spacing constant (40px)
- Verify canvas dimensions match constants

### Debug Tips
```typescript
// Add to event handlers
console.log('Strokes:', this.strokes.length);
console.log('Preview state:', this.previewState);
console.log('Pan offset:', this.panOffsetX);
console.log('Eraser mode:', this.isEraserMode);
```

## Development Workflow

### Adding New Features
1. Update state in `math-drawer.tsx` with `@State()` decorator
2. Add UI elements in `render()` method
3. Implement event handlers
4. Add styling in `math-drawer.css`
5. Update canvas operations in `stroke-manager.ts` if needed
6. Test with Playwright MCP
7. Update backlog.md with completion status

### Modifying Canvas Behavior
1. Most canvas logic is in `stroke-manager.ts`
2. Drawing functions: `drawStroke()`, `drawGrid()`, `clearCanvas()`
3. Always account for `panOffsetX` in coordinate calculations
4. Use `SCALE_FACTOR` for high-DPI rendering

### Styling Guidelines
- Use CSS custom properties for theming (not currently implemented)
- Keep all styles in `math-drawer.css`
- Follow BEM-like naming: `.inline-result.outdated`
- Use flexbox for layout, avoid floats

## Browser Compatibility
- Chrome 70+
- Firefox 65+
- Safari 12+
- Edge 79+
- iOS Safari 12+
- Android Chrome 70+

## Performance Considerations
- Canvas redraw on every stroke point (60fps target)
- Stroke history grows unbounded (consider limiting)
- Grid rendering adds ~2ms per frame
- API calls throttled by user interaction (click to convert)

## Future Enhancements (from backlog.md)
- Component configuration via props
- Accessibility improvements (aria-labels, keyboard nav)
- Custom icon sets
- Theming system
- Documentation updates
