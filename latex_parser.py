"""
LaTeX to SAN Hybrid Tree Label Converter
Converts LaTeX mathematical expressions to the hybrid tree format expected by SAN decoder.
"""

import re
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class TreeNode:
    """Represents a node in the mathematical expression tree"""
    token: str
    node_id: int
    parent_id: int
    children: List['TreeNode'] = None
    relation: str = 'right'  # Spatial relationship to parent
    structure_flags: List[bool] = None  # 7 structure flags
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.structure_flags is None:
            self.structure_flags = [False] * 7


class LaTeXTokenizer:
    """Tokenizes LaTeX mathematical expressions into semantic tokens"""
    
    def __init__(self):
        # Define LaTeX command patterns in order of precedence
        self.command_patterns = [
            # Multi-character functions and operators
            (r'\\frac\{([^{}]*)\}\{([^{}]*)\}', self._handle_frac),
            (r'\\sqrt\{([^{}]*)\}', self._handle_sqrt),
            (r'\\sqrt\[([^\]]*)\]\{([^{}]*)\}', self._handle_nth_root),
            (r'\\sum_\{([^{}]*)\}\^\{([^{}]*)\}', self._handle_sum_limits),
            (r'\\int_\{([^{}]*)\}\^\{([^{}]*)\}', self._handle_int_limits),
            (r'\\lim_\{([^{}]*)\}', self._handle_lim),
            
            # Subscripts and superscripts
            (r'_\{([^{}]*)\}', self._handle_subscript),
            (r'\^\{([^{}]*)\}', self._handle_superscript),
            (r'_([a-zA-Z0-9])', self._handle_simple_subscript),
            (r'\^([a-zA-Z0-9])', self._handle_simple_superscript),
            
            # Special commands
            (r'\\(sin|cos|tan|log|ln|exp|lim|sum|int|prod)', self._handle_function),
            (r'\\(alpha|beta|gamma|delta|theta|phi|pi|sigma|mu|lambda|Delta)', self._handle_greek),
            (r'\\(pm|times|div|cdot|leq|geq|neq|in|infty|rightarrow|ldots|cdots)', self._handle_symbol),
            (r'\\(left|right)([()[\]|])', self._handle_delimiter),
            
            # Brackets and delimiters
            (r'\{([^{}]*)\}', self._handle_group),
            
            # Single characters and numbers
            (r'[a-zA-Z]', self._handle_variable),
            (r'[0-9]+', self._handle_number),
            (r'[+\-=<>()[\]|,./!]', self._handle_operator),
        ]
    
    def tokenize(self, latex_string: str) -> List[Dict]:
        """
        Tokenize LaTeX string into structured tokens
        Returns list of token dictionaries with type, value, and structure info
        """
        tokens = []
        pos = 0
        
        # Clean input
        latex_string = latex_string.strip()
        
        while pos < len(latex_string):
            if latex_string[pos].isspace():
                pos += 1
                continue
                
            matched = False
            
            # Try each pattern
            for pattern, handler in self.command_patterns:
                regex = re.compile(pattern)
                match = regex.match(latex_string, pos)
                
                if match:
                    new_tokens = handler(match)
                    tokens.extend(new_tokens)
                    pos = match.end()
                    matched = True
                    break
            
            if not matched:
                # Unknown character - treat as literal
                tokens.append({
                    'type': 'literal',
                    'value': latex_string[pos],
                    'structure': 'none'
                })
                pos += 1
        
        return tokens
    
    # Token handlers
    def _handle_frac(self, match) -> List[Dict]:
        """Handle \frac{num}{den}"""
        numerator = match.group(1)
        denominator = match.group(2)
        
        return [
            {'type': 'command', 'value': '\\frac', 'structure': 'above'},
            {'type': 'group', 'value': numerator, 'structure': 'above'},
            {'type': 'group', 'value': denominator, 'structure': 'below'}
        ]
    
    def _handle_sqrt(self, match) -> List[Dict]:
        """Handle \sqrt{content}"""
        content = match.group(1)
        return [
            {'type': 'command', 'value': '\\sqrt', 'structure': 'above'},
            {'type': 'group', 'value': content, 'structure': 'inside'}
        ]
    
    def _handle_nth_root(self, match) -> List[Dict]:
        """Handle \sqrt[n]{content}"""
        index = match.group(1)
        content = match.group(2)
        return [
            {'type': 'command', 'value': '\\sqrt', 'structure': 'above'},
            {'type': 'group', 'value': index, 'structure': 'L-sup'},
            {'type': 'group', 'value': content, 'structure': 'inside'}
        ]
    
    def _handle_subscript(self, match) -> List[Dict]:
        """Handle _{content}"""
        content = match.group(1)
        return [{'type': 'group', 'value': content, 'structure': 'sub'}]
    
    def _handle_superscript(self, match) -> List[Dict]:
        """Handle ^{content}"""
        content = match.group(1)
        return [{'type': 'group', 'value': content, 'structure': 'sup'}]
    
    def _handle_simple_subscript(self, match) -> List[Dict]:
        """Handle _x"""
        content = match.group(1)
        return [{'type': 'variable', 'value': content, 'structure': 'sub'}]
    
    def _handle_simple_superscript(self, match) -> List[Dict]:
        """Handle ^x"""
        content = match.group(1)
        return [{'type': 'variable', 'value': content, 'structure': 'sup'}]
    
    def _handle_sum_limits(self, match) -> List[Dict]:
        """Handle \sum_{lower}^{upper}"""
        lower = match.group(1)
        upper = match.group(2)
        return [
            {'type': 'command', 'value': '\\sum', 'structure': 'none'},
            {'type': 'group', 'value': lower, 'structure': 'sub'},
            {'type': 'group', 'value': upper, 'structure': 'sup'}
        ]
    
    def _handle_int_limits(self, match) -> List[Dict]:
        """Handle \int_{lower}^{upper}"""
        lower = match.group(1)
        upper = match.group(2)
        return [
            {'type': 'command', 'value': '\\int', 'structure': 'none'},
            {'type': 'group', 'value': lower, 'structure': 'sub'},
            {'type': 'group', 'value': upper, 'structure': 'sup'}
        ]
    
    def _handle_lim(self, match) -> List[Dict]:
        """Handle \lim_{content}"""
        content = match.group(1)
        return [
            {'type': 'command', 'value': '\\lim', 'structure': 'none'},
            {'type': 'group', 'value': content, 'structure': 'sub'}
        ]
    
    def _handle_function(self, match) -> List[Dict]:
        """Handle function names like \sin, \cos"""
        func_name = match.group(1)
        return [{'type': 'function', 'value': f'\\{func_name}', 'structure': 'none'}]
    
    def _handle_greek(self, match) -> List[Dict]:
        """Handle Greek letters"""
        letter = match.group(1)
        return [{'type': 'greek', 'value': f'\\{letter}', 'structure': 'none'}]
    
    def _handle_symbol(self, match) -> List[Dict]:
        """Handle special symbols"""
        symbol = match.group(1)
        return [{'type': 'symbol', 'value': f'\\{symbol}', 'structure': 'none'}]
    
    def _handle_delimiter(self, match) -> List[Dict]:
        """Handle \left( \right) etc."""
        side = match.group(1)
        delim = match.group(2)
        return [{'type': 'delimiter', 'value': delim, 'structure': 'none'}]
    
    def _handle_group(self, match) -> List[Dict]:
        """Handle {content} - recursively tokenize content"""
        content = match.group(1)
        if content:
            # Recursively tokenize group content
            subtokens = self.tokenize(content)
            return [{'type': 'group_start', 'value': '{', 'structure': 'none'}] + \
                   subtokens + \
                   [{'type': 'group_end', 'value': '}', 'structure': 'none'}]
        else:
            return []
    
    def _handle_variable(self, match) -> List[Dict]:
        """Handle single variables"""
        var = match.group(0)
        return [{'type': 'variable', 'value': var, 'structure': 'none'}]
    
    def _handle_number(self, match) -> List[Dict]:
        """Handle numbers"""
        num = match.group(0)
        return [{'type': 'number', 'value': num, 'structure': 'none'}]
    
    def _handle_operator(self, match) -> List[Dict]:
        """Handle operators and punctuation"""
        op = match.group(0)
        return [{'type': 'operator', 'value': op, 'structure': 'none'}]


class HybridTreeBuilder:
    """Builds hierarchical tree structure from tokenized LaTeX"""
    
    def __init__(self, vocab_dict: Dict[str, int]):
        self.vocab = vocab_dict
        self.structure_relations = {
            'none': 'right',
            'sub': 'sub', 
            'sup': 'sup',
            'above': 'above',
            'below': 'below',
            'inside': 'inside',
            'L-sup': 'L-sup'
        }
        
        # Map structure types to flag indices (7 flags total)
        self.structure_flags = {
            'right': [True, False, False, False, False, False, False],
            'sub': [False, True, False, False, False, False, False],
            'sup': [False, False, True, False, False, False, False],
            'above': [False, False, False, True, False, False, False],
            'below': [False, False, False, False, True, False, False],
            'inside': [False, False, False, False, False, True, False],
            'L-sup': [False, False, False, False, False, False, True]
        }
    
    def build_tree(self, tokens: List[Dict]) -> List[TreeNode]:
        """Convert tokens to tree structure"""
        nodes = []
        node_id = 0
        
        # Add start token
        nodes.append(TreeNode(
            token='<sos>',
            node_id=node_id,
            parent_id=-1,
            relation='right',
            structure_flags=self.structure_flags['right']
        ))
        node_id += 1
        
        # Process tokens
        parent_stack = [0]  # Stack of parent node IDs
        
        for token_info in tokens:
            token_value = token_info['value']
            structure = token_info.get('structure', 'none')
            
            # Handle groups recursively
            if token_info['type'] == 'group':
                group_tokens = LaTeXTokenizer().tokenize(token_value)
                group_nodes = self._process_token_group(
                    group_tokens, node_id, parent_stack[-1], structure
                )
                nodes.extend(group_nodes)
                node_id += len(group_nodes)
                continue
            
            # Skip group delimiters in flat representation
            if token_info['type'] in ['group_start', 'group_end']:
                continue
            
            # Create node
            parent_id = parent_stack[-1]
            relation = self.structure_relations.get(structure, 'right')
            
            node = TreeNode(
                token=token_value,
                node_id=node_id,
                parent_id=parent_id,
                relation=relation,
                structure_flags=self.structure_flags.get(relation, self.structure_flags['right'])
            )
            
            nodes.append(node)
            
            # Update parent stack for structural relationships
            if structure in ['sub', 'sup', 'above', 'below', 'inside', 'L-sup']:
                # These create child relationships
                parent_stack.append(node_id)
            else:
                # Regular right progression
                if len(parent_stack) > 1 and structure == 'none':
                    parent_stack.pop()  # Return to previous level
            
            node_id += 1
        
        # Add end token
        nodes.append(TreeNode(
            token='<eos>',
            node_id=node_id,
            parent_id=parent_stack[-1],
            relation='right',
            structure_flags=self.structure_flags['right']
        ))
        
        return nodes
    
    def _process_token_group(self, tokens: List[Dict], start_id: int, 
                           parent_id: int, structure: str) -> List[TreeNode]:
        """Process a group of tokens with specific structure"""
        nodes = []
        node_id = start_id
        
        for token_info in tokens:
            token_value = token_info['value']
            relation = self.structure_relations.get(structure, 'right')
            
            node = TreeNode(
                token=token_value,
                node_id=node_id,
                parent_id=parent_id,
                relation=relation,
                structure_flags=self.structure_flags.get(relation, self.structure_flags['right'])
            )
            
            nodes.append(node)
            node_id += 1
            parent_id = node_id - 1  # Sequential within group
        
        return nodes


class SANLabelEncoder:
    """Converts tree structure to SAN hybrid tree tensor format"""
    
    def __init__(self, vocab_dict: Dict[str, int], max_length: int = 50):
        self.vocab = vocab_dict
        self.max_length = max_length
        self.relation_vocab = {
            'right': 0,
            'sub': 1, 
            'sup': 2,
            'above': 3,
            'below': 4,
            'inside': 5,
            'L-sup': 6
        }
    
    def encode_tree(self, nodes: List[TreeNode]) -> torch.Tensor:
        """
        Convert tree nodes to SAN label tensor format
        
        Returns:
            labels: [max_length, 11] tensor
                [:, 0]: step index
                [:, 1]: word token ID  
                [:, 2]: parent step ID
                [:, 3]: relation token ID
                [:, 4:11]: structure flags (7 flags)
        """
        # Truncate or pad to max_length
        num_nodes = min(len(nodes), self.max_length)
        labels = torch.zeros(self.max_length, 11, dtype=torch.long)
        
        for i in range(num_nodes):
            node = nodes[i]
            
            # Step index
            labels[i, 0] = i
            
            # Word token ID (from vocabulary)
            token_id = self.vocab.get(node.token, self.vocab.get('<eos>', 0))
            labels[i, 1] = token_id
            
            # Parent step ID
            parent_id = max(0, min(node.parent_id, self.max_length - 1))
            labels[i, 2] = parent_id
            
            # Relation token ID
            relation_id = self.relation_vocab.get(node.relation, 0)
            labels[i, 3] = relation_id
            
            # Structure flags
            for j, flag in enumerate(node.structure_flags):
                labels[i, 4 + j] = 1 if flag else 0
        
        return labels


class LaTeXToSANConverter:
    """Complete LaTeX to SAN label converter"""
    
    def __init__(self, word_path: str, max_length: int = 50):
        # Load vocabulary
        self.vocab_dict = self._load_vocabulary(word_path)
        self.max_length = max_length
        
        # Initialize components
        self.tokenizer = LaTeXTokenizer()
        self.tree_builder = HybridTreeBuilder(self.vocab_dict)
        self.encoder = SANLabelEncoder(self.vocab_dict, max_length)
    
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
            print(f"Warning: Vocabulary file {word_path} not found")
            # Create minimal vocabulary
            vocab = {'<eos>': 0, '<sos>': 1, 'struct': 2}
        
        return vocab
    
    def convert(self, latex_string: str) -> torch.Tensor:
        """
        Convert LaTeX string to SAN label tensor
        
        Args:
            latex_string: LaTeX mathematical expression
            
        Returns:
            labels: [max_length, 11] tensor in SAN format
        """
        try:
            # Step 1: Tokenize LaTeX
            tokens = self.tokenizer.tokenize(latex_string)
            
            # Step 2: Build tree structure  
            tree_nodes = self.tree_builder.build_tree(tokens)
            
            # Step 3: Encode to tensor format
            labels = self.encoder.encode_tree(tree_nodes)
            
            return labels
            
        except Exception as e:
            print(f"Warning: Failed to convert LaTeX '{latex_string}': {e}")
            # Return dummy tensor with just start/end tokens
            labels = torch.zeros(self.max_length, 11, dtype=torch.long)
            labels[0, 1] = self.vocab_dict.get('<sos>', 1)  # Start token
            labels[1, 1] = self.vocab_dict.get('<eos>', 0)  # End token
            labels[1, 2] = 0  # Parent of end is start
            return labels
    
    def batch_convert(self, latex_strings: List[str]) -> torch.Tensor:
        """Convert batch of LaTeX strings"""
        batch_labels = []
        for latex_str in latex_strings:
            labels = self.convert(latex_str)
            batch_labels.append(labels)
        
        return torch.stack(batch_labels)


# Test and validation functions
def test_latex_converter():
    """Test the LaTeX converter with sample expressions"""
    converter = LaTeXToSANConverter('data/word.txt')
    
    # Test cases
    test_cases = [
        "O_{ij}",
        "v_{i}=C_{i}-V", 
        "\\frac{a}{b}",
        "\\sqrt{x}",
        "\\sum_{i=1}^{n} x_i",
        "\\int_0^1 f(x) dx",
        "\\alpha + \\beta",
        "x^2 + y^2 = 1"
    ]
    
    print("Testing LaTeX to SAN conversion:")
    print("=" * 50)
    
    for latex_str in test_cases:
        labels = converter.convert(latex_str)
        print(f"LaTeX: {latex_str}")
        print(f"Label shape: {labels.shape}")
        print(f"Non-zero tokens: {(labels[:, 1] > 0).sum()}")
        print(f"Token IDs: {labels[:5, 1].tolist()}")  # First 5 tokens
        print("-" * 30)
    
    return converter


if __name__ == '__main__':
    # Run tests
    converter = test_latex_converter()
    
    # Test batch conversion
    batch_latex = ["O_{ij}", "v_i=C_i-V", "\\frac{1}{2}"]
    batch_labels = converter.batch_convert(batch_latex)
    print(f"Batch shape: {batch_labels.shape}")
    print("LaTeX to SAN converter ready!")