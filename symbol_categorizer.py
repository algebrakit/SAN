"""
Symbol Categorizer Module

This module categorizes mathematical symbols from the handwriting dataset 
into logical groups for analysis.
"""

import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple


class SymbolCategorizer:
    """Categorizes mathematical symbols into logical groups."""
    
    def __init__(self):
        """Initialize the categorizer with predefined categories."""
        self.categories = {
            'Numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            
            'Uppercase Letters': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
            
            'Lowercase Letters': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                                 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
            
            'Greek Lowercase': ['\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\zeta', '\\eta', 
                               '\\theta', '\\iota', '\\kappa', '\\lambda', '\\mu', '\\nu', '\\xi', '\\omicron', 
                               '\\pi', '\\rho', '\\sigma', '\\tau', '\\upsilon', '\\phi', '\\chi', '\\psi', 
                               '\\omega', '\\vartheta', '\\varphi', '\\varsigma', '\\varpi'],
            
            'Greek Uppercase': ['\\Alpha', '\\Beta', '\\Gamma', '\\Delta', '\\Epsilon', '\\Zeta', '\\Eta', 
                               '\\Theta', '\\Iota', '\\Kappa', '\\Lambda', '\\Mu', '\\Nu', '\\Xi', '\\Omicron', 
                               '\\Pi', '\\Rho', '\\Sigma', '\\Tau', '\\Upsilon', '\\Phi', '\\Chi', '\\Psi', 
                               '\\Omega'],
            
            'Basic Operators': ['+', '-', '*', '/', '=', '<', '>', '\\pm', '\\mp', '\\times', '\\div', 
                               '\\cdot', '\\circ', '\\bullet', '\\ne', '\\le', '\\ge', '\\ll', '\\gg'],
            
            'Set Theory': ['\\in', '\\notin', '\\subset', '\\supset', '\\subseteq', '\\supseteq', 
                          '\\subsetneq', '\\cup', '\\cap', '\\emptyset', '\\ni'],
            
            'Logic & Quantifiers': ['\\forall', '\\exists', '\\neg', '\\wedge', '\\vee', '\\rightarrow', 
                                   '\\leftarrow', '\\leftrightarrow', '\\Rightarrow', '\\Leftrightarrow', 
                                   '\\iff', '\\models', '\\vdash', '\\Vdash'],
            
            'Brackets & Delimiters': ['(', ')', '[', ']', '\\{', '\\}', '|', '\\|', '\\langle', '\\rangle', 
                                     '\\lfloor', '\\rfloor', '\\lceil', '\\rceil'],
            
            'Relations & Equivalence': ['\\equiv', '\\cong', '\\simeq', '\\approx', '\\sim', '\\propto', 
                                       '\\triangleq'],
            
            'Calculus & Analysis': ['\\partial', '\\nabla', '\\int', '\\oint', '\\iint', '\\sum', '\\prod', 
                                   '\\infty', '\\limit'],
            
            'Functions & Arrows': ['\\mapsto', '\\hookrightarrow', '\\longrightarrow', '\\rightleftharpoons'],
            
            'Accents & Modifiers': ['\\hat', '\\tilde', '\\vec', '\\overline', '\\underline', '\\dot', 
                                    '\\prime', '\\hbar'],
            
            'Special Functions': ['\\sqrt', '\\frac'],
            
            'Blackboard Bold': ['\\mathbb{A}', '\\mathbb{C}', '\\mathbb{D}', '\\mathbb{E}', '\\mathbb{F}', 
                               '\\mathbb{I}', '\\mathbb{K}', '\\mathbb{L}', '\\mathbb{N}', '\\mathbb{P}', 
                               '\\mathbb{Q}', '\\mathbb{R}', '\\mathbb{S}', '\\mathbb{T}', '\\mathbb{W}', 
                               '\\mathbb{X}', '\\mathbb{Z}'],
            
            'Big Operators': ['\\bigcap', '\\bigcup', '\\bigwedge', '\\bigvee', '\\bigoplus'],
            
            'Special Symbols': ['\\aleph', '\\angle', '\\top', '\\perp', '\\dagger', '\\#', '\\%', '\\&'],
            
            'Binary Operations': ['\\oplus', '\\ominus', '\\otimes', '\\odot', '\\backslash'],
            
            'Punctuation & Misc': ['.', ',', ':', ';', '!', '?', '\\vdots']
        }
    
    def categorize_symbols(self, symbols_data: pd.DataFrame) -> Dict[str, List[Tuple[str, int, float]]]:
        """
        Categorize symbols from the dataset.
        
        Args:
            symbols_data (pd.DataFrame): DataFrame with Symbol, Count, Percentage columns
            
        Returns:
            Dict[str, List[Tuple[str, int, float]]]: Dictionary mapping category to list of (symbol, count, percentage)
        """
        categorized = defaultdict(list)
        uncategorized = []
        
        for _, row in symbols_data.iterrows():
            symbol = row['Symbol']
            count = row['Count']
            percentage = row['Percentage']
            
            found_category = None
            for category, symbol_list in self.categories.items():
                if symbol in symbol_list:
                    categorized[category].append((symbol, count, percentage))
                    found_category = category
                    break
            
            if found_category is None:
                uncategorized.append((symbol, count, percentage))
        
        # Add uncategorized symbols
        if uncategorized:
            categorized['Uncategorized'] = uncategorized
        
        return dict(categorized)
    
    def print_categorized_analysis(self, symbols_data: pd.DataFrame):
        """Print a detailed categorized analysis of all symbols."""
        categorized = self.categorize_symbols(symbols_data)
        
        print("=" * 80)
        print("COMPLETE CATEGORIZED SYMBOL ANALYSIS")
        print("=" * 80)
        print(f"Total unique symbols: {len(symbols_data)}")
        print(f"Categories found: {len(categorized)}")
        
        total_samples = symbols_data['Count'].sum()
        
        for category, symbols in categorized.items():
            category_count = len(symbols)
            category_samples = sum(count for _, count, _ in symbols)
            category_percentage = (category_samples / total_samples) * 100
            
            print(f"\n{'-' * 60}")
            print(f"{category.upper()} ({category_count} symbols, {category_samples} samples, {category_percentage:.1f}%)")
            print(f"{'-' * 60}")
            
            # Sort by count (descending)
            sorted_symbols = sorted(symbols, key=lambda x: x[1], reverse=True)
            
            for symbol, count, percentage in sorted_symbols:
                print(f"  {symbol:20} : {count:4d} samples ({percentage:5.2f}%)")
    
    def get_category_summary(self, symbols_data: pd.DataFrame) -> pd.DataFrame:
        """
        Get a summary DataFrame of categories.
        
        Returns:
            pd.DataFrame: Summary with category statistics
        """
        categorized = self.categorize_symbols(symbols_data)
        total_samples = symbols_data['Count'].sum()
        
        summary_data = []
        for category, symbols in categorized.items():
            category_count = len(symbols)
            category_samples = sum(count for _, count, _ in symbols)
            category_percentage = (category_samples / total_samples) * 100
            avg_samples = category_samples / category_count if category_count > 0 else 0
            
            summary_data.append({
                'Category': category,
                'Number of Symbols': category_count,
                'Total Samples': category_samples,
                'Percentage of Dataset': category_percentage,
                'Average Samples per Symbol': avg_samples
            })
        
        return pd.DataFrame(summary_data).sort_values('Total Samples', ascending=False)


def main():
    """Main function to run the categorization analysis."""
    # Load the data
    df = pd.read_csv('symbol_distribution_analysis.csv')
    
    # Create categorizer and run analysis
    categorizer = SymbolCategorizer()
    categorizer.print_categorized_analysis(df)
    
    print("\n" + "=" * 80)
    print("CATEGORY SUMMARY")
    print("=" * 80)
    
    summary = categorizer.get_category_summary(df)
    print(summary.to_string(index=False, float_format='{:.2f}'.format))


if __name__ == "__main__":
    main()