#!/usr/bin/env python3
"""
Debug script to analyze how \lambda_{2(M)} is being parsed
"""
from simple_latex_converter import SimpleLaTeXConverter

def debug_lambda_parsing():
    print("=== DEBUGGING LAMBDA PARSING ===")
    
    # Create converter
    converter = SimpleLaTeXConverter("data/word.txt", 20)
    
    # Test the problematic expression
    test_expr = r"\lambda_{2(M)}"
    
    print(f"\nTesting expression: '{test_expr}'")
    print(f"Length: {len(test_expr)} characters")
    print(f"Characters: {list(test_expr)}")
    
    # Check if full expression is in vocabulary
    print(f"\nFull expression in vocabulary: {test_expr in converter.vocab_dict}")
    
    # Check individual characters
    print("\nIndividual character analysis:")
    for i, char in enumerate(test_expr):
        in_vocab = char in converter.vocab_dict
        token_id = converter.vocab_dict.get(char, -1)
        print(f"  {i}: '{char}' -> in vocab: {in_vocab}, token: {token_id}")
    
    # Show what the converter generates
    print(f"\n=== CONVERSION RESULT ===")
    labels = converter.debug_conversion(test_expr)
    
    # Analyze the pattern
    print(f"\n=== PATTERN ANALYSIS ===")
    vocab_reverse = {v: k for k, v in converter.vocab_dict.items()}
    
    sequence = []
    for i in range(20):
        token = labels[i, 0].item()
        symbol = vocab_reverse.get(token, f"UNK_{token}")
        sequence.append(symbol)
        if symbol == '<eos>':
            break
    
    print(f"Generated sequence: {sequence}")
    print(f"As string: {''.join([s for s in sequence if s != '<eos>'])}")
    
    # Check for LaTeX commands in vocabulary
    print(f"\n=== LATEX COMMAND ANALYSIS ===")
    latex_commands = [key for key in converter.vocab_dict.keys() if key.startswith('\\')]
    print(f"LaTeX commands in vocabulary ({len(latex_commands)}):")
    for cmd in sorted(latex_commands)[:10]:  # Show first 10
        print(f"  '{cmd}' -> token {converter.vocab_dict[cmd]}")
    if len(latex_commands) > 10:
        print(f"  ... and {len(latex_commands) - 10} more")
    
    # Specifically check for lambda
    lambda_variants = ['\\lambda', 'lambda', 'λ']
    print(f"\nLambda variants in vocabulary:")
    for variant in lambda_variants:
        in_vocab = variant in converter.vocab_dict
        token_id = converter.vocab_dict.get(variant, -1)
        print(f"  '{variant}' -> in vocab: {in_vocab}, token: {token_id}")

if __name__ == "__main__":
    debug_lambda_parsing()