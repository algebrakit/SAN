export type OffsetType = 'DISABLE' | 'STRONG_PENALIZE' | 'PENALIZE' | 'BOOST' | 'STRONG_BOOST';

export interface SymbolAdjustment {
  symbol: string;
  offset: OffsetType;
}

export const OFFSET_OPTIONS: OffsetType[] = [
  'DISABLE',
  'STRONG_PENALIZE',
  'PENALIZE',
  'BOOST',
  'STRONG_BOOST'
];
