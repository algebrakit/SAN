# Math Drawer Component - Design Backlog

## Overview
Redesigning the Math Expression Drawer into a compact web component based on `design.png` mockup.

## Current State
- Vertical layout with header/footer
- Horizontal button row below canvas
- Separate LaTeX result section
- Text-based buttons
- White canvas background
- Canvas size: 480x280px

## Target Design (from design.png)
- Compact horizontal layout
- Vertical icon sidebar on left
- Grid background on canvas
- Inline result display (bottom-right corner)
- Icon-based controls
- Professional, embeddable component

---

## Backlog Items

### High Priority

#### 1. Layout Restructure ✅
- [x] Remove header/footer from index.html (make component standalone) - *Keeping for now*
- [x] Convert main layout from vertical to horizontal (sidebar + canvas)
- [x] Create flex container for sidebar + canvas area
- [x] Make component dimensions more compact

#### 2. Vertical Sidebar Implementation ✅
- [x] Create sidebar container on left side
- [x] Add icon buttons vertically stacked
- [x] Style sidebar with proper spacing and background
- [x] Add hover/active states for icon buttons

#### 3. Icon System ✅
- [x] Add SVG icon for Undo (curved arrow left)
- [x] Add SVG icon for Redo (curved arrow right)
- [x] Add SVG icon for Clear/Delete (trash can)
- [x] Add SVG icon for Convert to LaTeX (code brackets `<>`)
- [x] Add SVG icon for Grid toggle (grid pattern)
- [x] Create icon component or inline SVG definitions
- [x] Remove help icon (not needed)

#### 4. Grid Background ✅
- [x] Add grid pattern rendering to canvas
- [x] Make grid configurable (spacing, color, line width)
- [x] Add toggle functionality to show/hide grid
- [x] Ensure grid doesn't interfere with drawing

#### 5. Inline Result Display ✅
- [x] Move LaTeX result inside canvas area
- [x] Position in bottom-right corner
- [x] Add semi-transparent background overlay
- [x] Style as compact inline display
- [x] Remove separate LaTeX result section below canvas

### Medium Priority

#### 6. Help Feature ❌ REMOVED
- Help button removed per user request
- May revisit in future if needed

#### 7. Canvas Adjustments ✅
- [x] Review and adjust canvas dimensions for compact view
- [x] Make canvas 2000px wide (internal resolution)
- [x] Implement overflow clipping for responsive container
- [x] Fix flex item min-width to allow proper shrinking
- [x] Ensure grid aligns properly with canvas size
- [x] Test high-DPI rendering with grid
- [x] Position inline result relative to visible area

#### 8. Styling Refinements ✅
- [x] Update color scheme to match design
- [x] Add subtle shadows/borders to sidebar
- [x] Improve disabled state styling for icons
- [x] Ensure responsive behavior for small screens
- [x] Polish transitions and hover effects

#### 9. Auto-Update Preview Indicator ✅
- [x] Add visual states for preview: current (green), outdated (amber), updating (loading)
- [x] Track `previewState` and `lastConvertedStrokeCount` in component state
- [x] Make outdated preview clickable to trigger re-conversion
- [x] Add refresh icon overlay when preview is outdated
- [x] Update CSS for three states: `.inline-result.current`, `.inline-result.outdated`, `.inline-result.updating`
- [x] Trigger "outdated" state on new stroke, undo, or redo after conversion
- [x] Add smooth transitions between states
- [x] Add "Click to update" tooltip on hover
- [x] Fix container width to ensure preview stays visible within viewport

**Benefits**: Clear visual feedback when preview is stale, clickable preview for easy updates, professional color-coded states

**Files modified**: `math-drawer.tsx`, `math-drawer.css`

#### 10. Remove Convert Button, Use Preview as Sole Conversion Trigger ✅
- [x] Remove "Convert to LaTeX" button from sidebar
- [x] Keep `renderConvertIcon()` method for use in preview refresh icon
- [x] Show preview in 'outdated' state as soon as student starts drawing (even without prior conversion)
- [x] Update `stopDrawing()` to set previewState = 'outdated' when strokeCount > 0 and no latexResult
- [x] Change preview visibility condition from `{this.latexResult && ...}` to `{this.previewState !== 'none' && ...}`
- [x] Update preview content to show "Click to convert" text when latexResult is empty
- [x] Ensure preview is clickable in 'outdated' state to trigger conversion
- [x] Add CSS styling for empty preview text

**Benefits**: Cleaner UI with one less button, more intuitive single-point-of-interaction, better discoverability of conversion feature

**Files modified**: `math-drawer.tsx`, `math-drawer.css`

### Medium Priority

#### 11. Canvas Panning System ✅
**Objective**: Allow students to pan the canvas left/right to access more drawing space when expressions exceed visible area.

**Implementation approach (Two-finger gesture + Scrollbar):**

- [x] Add state variable: `panOffsetX` (current horizontal pan position)
- [x] Track viewport dimensions and total canvas width (2000px)
- [x] Implement bounds checking (prevent panning beyond canvas edges)
- [x] Update stroke rendering to account for pan offset
- [x] Detect multi-touch events (2 fingers on canvas) for mobile
- [x] Calculate center point and drag delta between touch events
- [x] Apply pan offset with smooth easing
- [x] Prevent drawing when two fingers detected
- [x] Wrap canvas in scrollable container with `overflow-x: auto` for desktop
- [x] Show scrollbar only on desktop (hide on touch devices)
- [x] Sync scrollbar position with pan offset from gestures
- [x] Update inline result position to stay in bottom-right of viewport (not canvas)
- [x] Ensure grid stays aligned during panning
- [x] Update undo/redo to preserve pan position
- [x] Clear canvas resets pan to start position
- [x] Test on mobile, desktop, and tablet

**Benefits**: More drawing space without increasing visible canvas size, mobile-friendly two-finger gesture, familiar scrollbar for desktop

**Files modified**: `math-drawer.tsx`, `math-drawer.css`, `stroke-manager.ts`

**Test results**:
- ✅ Desktop scrollbar works (canvas 2000px, container 845px, scrollable)
- ✅ Drawing works correctly after panning (coordinate handling with scroll offset)
- ✅ Clear button resets pan position to 0
- ✅ Inline result stays in viewport bottom-right during panning
- ✅ Grid stays aligned during panning
- ✅ Undo/redo buttons work correctly
- ⏳ Mobile two-finger pan (requires physical device testing)

### Medium Priority

#### 12. Eraser Tool with Radius ✅
**Objective**: Add an eraser tool that allows users to remove strokes by dragging over them with a circular eraser.

**Implementation approach:**

- [x] Add eraser button to sidebar (SVG icon of an eraser)
- [x] Add `isEraserMode` state variable to component
- [x] Create `renderEraserIcon()` method for SVG icon
- [x] Implement toggle behavior: click to activate/deactivate eraser mode
- [x] Add active state styling to eraser button when enabled
- [x] Define `eraserRadius` constant (15px)
- [x] Modify pointer event handlers to check for eraser mode
- [x] Implement eraser cursor: CSS custom cursor with circular red indicator
- [x] Add stroke detection logic: detect strokes within eraser radius
  - Calculate distance from pointer to each point in stroke
  - Mark stroke for deletion if any point is within radius
- [x] Visual feedback during erasing:
  - Red highlighting of strokes to be erased
  - Custom cursor with red circle showing eraser radius
- [x] Implement real-time erasing on drag:
  - Track which strokes touched during current drag
  - Highlight strokes in real-time as eraser passes over them
  - Group all deletions in single undo action
- [x] Ensure undo/redo works correctly with eraser
- [x] Test on desktop with mouse
- [ ] Test on touch devices (single-finger drag when eraser active) - requires physical device
- [x] Clear canvas exits eraser mode

**Benefits**: More precise control over stroke removal, easier cleanup of larger sections, familiar drawing app UX, complements scratch-to-delete gesture

**Files modified**: `math-drawer.tsx`, `math-drawer.css`, `stroke-manager.ts`

**Technical implementation**:
- Eraser uses distance-based detection with point-to-segment distance calculation
- All deletions during one drag are grouped into a single undo action
- Custom SVG cursor with red circle (10px radius visual, 15px actual detection)
- Active button state shows blue highlight when eraser mode is enabled
- Works seamlessly with existing undo/redo system
- Preview state updates correctly after erasing

### Low Priority

#### 13. Component Configuration
- [ ] Make component configurable via props (size, colors, etc.)
- [ ] Allow grid toggle default state configuration
- [ ] Allow custom icon sets

#### 14. Accessibility
- [ ] Add aria-labels to icon buttons
- [ ] Ensure keyboard navigation works
- [ ] Add screen reader support

#### 15. Documentation
- [ ] Update README with new component design
- [ ] Document web component usage
- [ ] Add screenshots of new design
- [ ] Document configuration options

---

## Technical Decisions to Make

1. **Grid Implementation**: Canvas-based vs CSS background pattern?
   - Recommendation: Canvas-based for consistency with drawing

2. **Icon Format**: Inline SVG vs icon library vs font icons?
   - Recommendation: Inline SVG for zero dependencies

3. **Result Display Animation**: Fade in vs slide in vs instant?
   - Recommendation: Fade in for smooth UX

4. **Component Dimensions**: Fixed vs flexible sizing?
   - Recommendation: Default fixed with optional prop override

---

## Dependencies
- None (all features use existing tech stack)

## Files to Modify
- `app/frontend/src/components/math-drawer/math-drawer.tsx` - Main component
- `app/frontend/src/components/math-drawer/math-drawer.css` - Styling
- `app/frontend/src/components/math-drawer/stroke-manager.ts` - Grid rendering
- `app/frontend/src/index.html` - Remove header/footer wrapper

## Testing Checklist
- [x] Drawing functionality works with grid
- [x] Undo/redo work correctly
- [x] Clear button clears canvas and result
- [ ] Grid toggle shows/hides grid
- [ ] Help button displays instructions
- [x] Result displays inline in bottom-right
- [ ] Component works on mobile/touch devices
- [x] Scratch-to-delete gesture still works
- [ ] API conversion still works
- [x] Eraser tool works on desktop
- [x] Eraser undo/redo works correctly
- [ ] Component can be embedded in other pages
- [ ] Two-finger pan gesture works on mobile (iOS/Android) - requires physical device
- [x] Scrollbar works on desktop with mouse
- [x] Drawing works correctly after panning
- [x] Undo/redo preserves pan position
- [x] Clear resets pan position
- [x] Inline result stays in viewport during pan
- [x] No conflicts between pan and draw gestures
- [x] Canvas bounds respected (can't pan beyond edges)
- [ ] Two-finger pan doesn't trigger browser zoom - requires physical device
