# akit-handwriting-canvas

A canvas component for capturing handwritten mathematical expressions and converting them to LaTeX.

## Features

- **Drawing**: Smooth, pressure-sensitive strokes with variable width
- **Eraser**: Toggle eraser mode to remove strokes by touching them
- **Scratch-to-delete**: Scribble over strokes to delete them
- **Undo/Redo**: Full history support
- **Panning**: Horizontal scrolling for wide expressions (desktop scrollbar, mobile two-finger gesture)
- **Grid**: Optional background grid for alignment
- **LaTeX conversion**: Send strokes to backend API and receive LaTeX output

## Usage

```html
<akit-handwriting-canvas></akit-handwriting-canvas>

<script>
  const canvas = document.querySelector('akit-handwriting-canvas');

  // Listen for LaTeX conversion results
  canvas.addEventListener('latexChanged', (e) => {
    console.log('LaTeX:', e.detail.latex);
  });

  // Programmatic control
  await canvas.convertToLatex();
  await canvas.undo();
  await canvas.redo();
  await canvas.clearCanvas();
</script>
```

## File Structure

| File | Purpose |
|------|---------|
| `akit-handwriting-canvas.tsx` | Main component (state, events, UI) |
| `stroke-manager.ts` | Stroke orchestration facade |
| `canvas-renderer.ts` | Drawing operations |
| `geometry.ts` | Math utilities |
| `eraser.ts` | Eraser detection |
| `api-service.ts` | Backend API calls |
| `icons.tsx` | SVG icons |
| `types.ts` | TypeScript interfaces |
| `config.ts` | All configuration constants |

<!-- Auto Generated Below -->


## Properties

| Property            | Attribute            | Description | Type                 | Default |
| ------------------- | -------------------- | ----------- | -------------------- | ------- |
| `showSubmitButton`  | `show-submit-button` |             | `boolean`            | `false` |
| `symbolAdjustments` | `symbol-adjustments` |             | `SymbolAdjustment[]` | `[]`    |


## Events

| Event          | Description | Type                              |
| -------------- | ----------- | --------------------------------- |
| `latexChanged` |             | `CustomEvent<{ latex: string; }>` |
| `submitted`    |             | `CustomEvent<{ latex: string; }>` |


## Methods

### `clearCanvas() => Promise<void>`

Clears the canvas without emitting latexChanged event. Use erase() to clear and emit.

#### Returns

Type: `Promise<void>`



### `convertToLatex() => Promise<void>`



#### Returns

Type: `Promise<void>`



### `getState() => Promise<HandwritingCanvasState>`



#### Returns

Type: `Promise<HandwritingCanvasState>`



### `redo() => Promise<void>`



#### Returns

Type: `Promise<void>`



### `restoreState(state: HandwritingCanvasState) => Promise<void>`



#### Parameters

| Name    | Type                     | Description |
| ------- | ------------------------ | ----------- |
| `state` | `HandwritingCanvasState` |             |

#### Returns

Type: `Promise<void>`



### `undo() => Promise<void>`



#### Returns

Type: `Promise<void>`




----------------------------------------------

*Built with [StencilJS](https://stenciljs.com/)*
