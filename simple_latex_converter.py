#!/usr/bin/env python3
"""
Simple LaTeX to SAN converter that actually works for basic symbols
This replaces the overly complex latex_parser.py for now
"""

import torch
from typing import Dict, List

class SimpleLaTeXConverter:
    """Simple converter that handles basic symbols correctly"""
    
    def __init__(self, word_path: str, max_length: int = 20):
        self.max_length = max_length
        self.vocab_dict = self._load_vocabulary(word_path)
        print(f"Loaded vocabulary with {len(self.vocab_dict)} symbols")
        
    def _load_vocabulary(self, word_path: str) -> Dict[str, int]:
        """Load word vocabulary from file"""
        vocab = {}
        try:
            with open(word_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    word = line.strip()
                    if word:
                        vocab[word] = idx
        except FileNotFoundError:
            print(f"Error: Vocabulary file {word_path} not found")
            raise
        return vocab
    
    def convert(self, latex_string: str) -> torch.Tensor:
        """
        Convert LaTeX string to SAN label tensor
        For now, handles simple single symbols correctly
        
        Returns:
            labels: [max_length, 11] tensor where:
                - Column 0: word token IDs  
                - Column 1: parent indices
                - Columns 2-10: structure flags (7 flags + 3 padding)
        """
        # Initialize output tensor
        labels = torch.zeros(self.max_length, 11, dtype=torch.long)
        
        # Clean input
        latex_string = latex_string.strip()
        
        if not latex_string:
            # Empty string - just end token
            labels[0, 0] = self.vocab_dict.get('<eos>', 0)
            return labels
        
        # Handle simple single character
        if len(latex_string) == 1 and latex_string in self.vocab_dict:
            # Create the target sequence (what model should predict)
            # Input sequence is: <sos>, L, <eos>
            # Target sequence should be: L, <eos>, <pad>
            
            # Position 0: target is the actual symbol (next after <sos> input)
            labels[0, 0] = self.vocab_dict[latex_string]
            labels[0, 1] = 0  # Parent is root
            
            # Position 1: target is end token (next after symbol input)
            labels[1, 0] = self.vocab_dict.get('<eos>', 0)
            labels[1, 1] = 0  # Parent is root
            
            # Position 2 and beyond: padding (token 0 = <eos> acts as padding)
            labels[2, 0] = self.vocab_dict.get('<eos>', 0)
            labels[2, 1] = 0
            
            return labels
        
        # Handle multi-character or unknown symbols
        if latex_string in self.vocab_dict:
            # Create the target sequence (what model should predict)
            # Input sequence is: <sos>, symbol, <eos>
            # Target sequence should be: symbol, <eos>, <pad>
            
            # Position 0: target is the symbol (next after <sos> input)
            labels[0, 0] = self.vocab_dict[latex_string]
            labels[0, 1] = 0  # Root
            
            # Position 1: target is end token (next after symbol input)
            labels[1, 0] = self.vocab_dict.get('<eos>', 0)
            labels[1, 1] = 0  # Root
            
            # Position 2 and beyond: padding
            labels[2, 0] = self.vocab_dict.get('<eos>', 0)
            labels[2, 1] = 0
            
            return labels
        
        # Unknown symbol - try to handle character by character
        print(f"Warning: Unknown symbol '{latex_string}', trying character-by-character")
        pos = 0
        
        # Build target sequence: each character, then <eos>, then padding
        # Note: We don't include <sos> in targets since it's the input
        
        # Process each character as targets
        for char in latex_string:
            if pos >= self.max_length - 1:  # Leave room for end token
                break
                
            if char in self.vocab_dict:
                labels[pos, 0] = self.vocab_dict[char]
                labels[pos, 1] = 0  # Parent is root
                pos += 1
            else:
                print(f"Warning: Character '{char}' not in vocabulary, skipping")
        
        # End token as final target
        if pos < self.max_length:
            labels[pos, 0] = self.vocab_dict.get('<eos>', 0)
            labels[pos, 1] = 0
            pos += 1
            
        # Fill remaining positions with padding (eos tokens)
        while pos < self.max_length:
            labels[pos, 0] = self.vocab_dict.get('<eos>', 0)
            labels[pos, 1] = 0
            pos += 1
        
        return labels
    
    def debug_conversion(self, latex_string: str):
        """Debug helper to show conversion process"""
        print(f"\n=== Debug Conversion: '{latex_string}' ===")
        labels = self.convert(latex_string)
        
        print(f"Input: '{latex_string}'")
        print(f"Output shape: {labels.shape}")
        
        # Decode first 5 positions
        vocab_reverse = {v: k for k, v in self.vocab_dict.items()}
        
        for i in range(min(5, self.max_length)):
            word_token = labels[i, 0].item()
            parent_idx = labels[i, 1].item()
            symbol = vocab_reverse.get(word_token, f"UNK_{word_token}")
            print(f"  Position {i}: token {word_token} = '{symbol}', parent = {parent_idx}")
            if symbol == '<eos>':
                break
        
        return labels


def test_converter():
    """Test the simple converter"""
    print("=== Testing Simple LaTeX Converter ===")
    
    converter = SimpleLaTeXConverter("data/word.txt", 20)
    
    test_cases = ['L', 'i', 'a', '2', '+', '=']
    
    for test_case in test_cases:
        converter.debug_conversion(test_case)


if __name__ == '__main__':
    test_converter()