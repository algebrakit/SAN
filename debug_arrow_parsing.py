#!/usr/bin/env python3
"""
Debug the parsing of A \to k(x) expression
"""

from smart_latex_converter import SmartLaTeXConverter
from dataset_stroke import Words

def debug_arrow_parsing():
    print("=== Debugging Arrow Expression Parsing ===")
    
    # Load vocabulary
    words = Words("data/word.txt")
    
    # Create smart converter
    converter = SmartLaTeXConverter("data/word.txt")
    
    # Test expression from inkml file
    test_expression = "A \\to k(x)"
    print(f"\nTest expression: '{test_expression}'")
    
    # Tokenize with smart converter
    tokens = converter.tokenize_latex(test_expression)
    print(f"\nTokenized by SmartLaTeXConverter: {tokens}")
    
    # Check each token in vocabulary
    print("\nToken analysis:")
    for i, token in enumerate(tokens):
        if token in converter.vocab_dict:
            token_id = converter.vocab_dict[token]
            print(f"  {i}: '{token}' -> token {token_id} OK")
        else:
            print(f"  {i}: '{token}' -> NOT IN VOCABULARY ERROR")
    
    # Test full conversion
    print(f"\n=== Full Label Processing ===")
    try:
        token_ids = converter.label_to_tokens(test_expression)
        print(f"Token IDs: {token_ids}")
        
        # Show each token ID mapping
        for i, token_id in enumerate(token_ids):
            if token_id in words.words_index_dict:
                symbol = words.words_index_dict[token_id]
                print(f"  {i}: token {token_id} -> '{symbol}'")
            else:
                print(f"  {i}: token {token_id} -> UNKNOWN")
                
    except Exception as e:
        print(f"Error in label conversion: {e}")

if __name__ == '__main__':
    debug_arrow_parsing()