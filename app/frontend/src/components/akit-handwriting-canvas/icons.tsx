import { h, FunctionalComponent } from '@stencil/core';

export const UndoIcon: FunctionalComponent = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 7v6h6"></path>
    <path d="M21 17a9 9 0 00-9-9 9 9 0 00-6 2.3L3 13"></path>
  </svg>
);

export const RedoIcon: FunctionalComponent = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 7v6h-6"></path>
    <path d="M3 17a9 9 0 019-9 9 9 0 016 2.3l3 2.7"></path>
  </svg>
);

export const TrashIcon: FunctionalComponent = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="3 6 5 6 21 6"></polyline>
    <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"></path>
  </svg>
);

export const EraserIcon: FunctionalComponent = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M20 20H7L3 16L12 7L17 12M11 11L13 13"></path>
    <path d="M8.5 15.5L6 13"></path>
  </svg>
);

export const SpinnerIcon: FunctionalComponent<{ class?: string }> = (props) => (
  <svg class={props.class || 'spinner'} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
  </svg>
);

export const CheckmarkIcon: FunctionalComponent<{ class?: string }> = (props) => (
  <svg class={props.class || 'checkmark'} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="20 6 9 17 4 12"></polyline>
  </svg>
);

export const SubmitIcon: FunctionalComponent<{ class?: string }> = (props) => (
  <svg class={props.class || 'submit-icon'} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"></line>
    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
  </svg>
);

export const UpArrowIcon: FunctionalComponent<{ class?: string }> = (props) => (
  <svg class={props.class || 'up-arrow'} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <line x1="12" y1="19" x2="12" y2="5"></line>
    <polyline points="5 12 12 5 19 12"></polyline>
  </svg>
);

export const KeyboardIcon: FunctionalComponent<{ class?: string }> = (props) => (
  <svg class={props.class || 'keyboard-icon'} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <rect x="2" y="4" width="20" height="16" rx="2" ry="2"></rect>
    <line x1="6" y1="8" x2="6" y2="8"></line>
    <line x1="10" y1="8" x2="10" y2="8"></line>
    <line x1="14" y1="8" x2="14" y2="8"></line>
    <line x1="18" y1="8" x2="18" y2="8"></line>
    <line x1="6" y1="12" x2="6" y2="12"></line>
    <line x1="10" y1="12" x2="10" y2="12"></line>
    <line x1="14" y1="12" x2="14" y2="12"></line>
    <line x1="18" y1="12" x2="18" y2="12"></line>
    <line x1="8" y1="16" x2="16" y2="16"></line>
  </svg>
);
