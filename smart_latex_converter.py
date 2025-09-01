#!/usr/bin/env python3
"""
Smart LaTeX converter that properly handles LaTeX commands as tokens
"""

import torch
import re
from typing import Dict, List, Tuple

class SmartLaTeXConverter:
    """Smart converter that tokenizes LaTeX commands properly"""
    
    def __init__(self, word_path: str, max_length: int = 20):
        self.max_length = max_length
        self.vocab_dict = self._load_vocabulary(word_path)
        self.latex_commands = self._extract_latex_commands()
        print(f"Loaded vocabulary with {len(self.vocab_dict)} symbols")
        print(f"Found {len(self.latex_commands)} LaTeX commands")
        
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
    
    def _extract_latex_commands(self) -> List[str]:
        """Extract LaTeX commands from vocabulary, sorted by length (longest first)"""
        commands = [word for word in self.vocab_dict.keys() if word.startswith('\\')]
        # Sort by length descending to match longest commands first
        return sorted(commands, key=len, reverse=True)
    
    def tokenize_latex(self, latex_string: str) -> List[str]:
        """
        Tokenize LaTeX string into proper tokens
        
        Example: \\lambda_{2(M)} -> ['\\lambda', '_', '{', '2', '(', 'M', ')', '}']
        """
        tokens = []
        i = 0
        
        while i < len(latex_string):
            # Try to match LaTeX commands first (longest match)
            matched = False
            for cmd in self.latex_commands:
                if latex_string[i:].startswith(cmd):
                    tokens.append(cmd)
                    i += len(cmd)
                    matched = True
                    break
            
            if not matched:
                # Single character
                char = latex_string[i]
                if char == ' ':
                    # Skip spaces - they're not tokens in LaTeX math expressions
                    pass
                elif char in self.vocab_dict:
                    tokens.append(char)
                else:
                    print(f"Warning: Character '{char}' not in vocabulary, skipping")
                i += 1
        
        return tokens
    
    def convert(self, latex_string: str) -> torch.Tensor:
        """
        Convert LaTeX string to SAN label tensor (compatible with SimpleLaTeXConverter)
        
        Returns:
            labels: [max_length, 11] tensor where:
                - Column 0: word token IDs  
                - Column 1: parent indices
                - Columns 2-10: structure flags (7 flags + 3 padding)
        """
        # Clean input
        latex_string = latex_string.strip()
        
        if not latex_string:
            # Empty string - just end token
            labels = torch.zeros(self.max_length, 11, dtype=torch.long)
            labels[0, 0] = self.vocab_dict.get('<eos>', 0)
            return labels
        
        # Tokenize the LaTeX string
        tokens = self.tokenize_latex(latex_string)
        
        if not tokens:
            print(f"Warning: No valid tokens found in '{latex_string}'")
            labels = torch.zeros(self.max_length, 11, dtype=torch.long)
            labels[0, 0] = self.vocab_dict.get('<eos>', 0)
            return labels
        
        # Check if we have too many tokens
        if len(tokens) >= self.max_length:
            print(f"Warning: Too many tokens ({len(tokens)}) for max length {self.max_length}")
            tokens = tokens[:self.max_length-1]  # Leave room for <eos>
        
        # Convert tokens to labels tensor [max_length, 11]
        labels = torch.zeros(self.max_length, 11, dtype=torch.long)
        
        pos = 0
        for token in tokens:
            if token in self.vocab_dict:
                labels[pos, 0] = self.vocab_dict[token]  # Word token in column 0
                labels[pos, 1] = 0  # Parent index (root)
                pos += 1
            else:
                print(f"Warning: Token '{token}' not in vocabulary, skipping")
        
        # Add end token
        if pos < self.max_length:
            labels[pos, 0] = self.vocab_dict.get('<eos>', 0)
            labels[pos, 1] = 0  # Parent index (root)
            pos += 1
        
        # Fill remaining positions with <eos> (padding)
        while pos < self.max_length:
            labels[pos, 0] = self.vocab_dict.get('<eos>', 0)
            labels[pos, 1] = 0  # Parent index (root)
            pos += 1
        
        return labels
    
    def label_to_tokens(self, latex_string: str) -> List[int]:
        """Convert LaTeX string to list of token IDs"""
        tokens = self.tokenize_latex(latex_string)
        token_ids = []
        
        for token in tokens:
            if token in self.vocab_dict:
                token_ids.append(self.vocab_dict[token])
            else:
                print(f"Warning: Token '{token}' not in vocabulary, skipping")
        
        # Add end token
        if '<eos>' in self.vocab_dict:
            token_ids.append(self.vocab_dict['<eos>'])
        
        return token_ids
    
    def debug_conversion(self, latex_string: str):
        """Debug helper to show conversion process"""
        print(f"\n=== Smart Debug Conversion: '{latex_string}' ===")
        
        # Show tokenization
        tokens = self.tokenize_latex(latex_string)
        print(f"Tokenized: {tokens}")
        
        # Show conversion
        labels = self.convert(latex_string)
        
        print(f"Labels shape: {labels.shape}")
        
        # Decode first positions
        vocab_reverse = {v: k for k, v in self.vocab_dict.items()}
        
        for i in range(min(len(tokens) + 2, self.max_length)):
            token_id = labels[i, 0].item()
            parent_idx = labels[i, 1].item()
            symbol = vocab_reverse.get(token_id, f"UNK_{token_id}")
            print(f"  Position {i}: token {token_id} = '{symbol}', parent = {parent_idx}")
            if symbol == '<eos>':
                break
        
        return labels


def test_smart_converter():
    """Test the smart converter"""
    print("=== Testing Smart LaTeX Converter ===")
    
    converter = SmartLaTeXConverter("data/word.txt", 20)
    
    test_cases = [
        r'\lambda_{2(M)}',
        r'\frac{1}{2}',
        r'\sqrt{x}',
        r'a=(x,v)',
        r'L',
        r'\alpha + \beta'
    ]
    
    for test_case in test_cases:
        converter.debug_conversion(test_case)


if __name__ == '__main__':
    test_smart_converter()