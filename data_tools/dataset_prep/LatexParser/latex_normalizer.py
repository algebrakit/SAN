#!/usr/bin/env python3
"""
LaTeX Normalizer for Mathematical Expression Recognition

This module provides functionality to normalize LaTeX expressions for
handwriting recognition training data. It parses LaTeX into a tree structure,
applies normalization rules, and reconstructs the normalized expression.
"""

from enum import Enum
from typing import List, Optional
from dataclasses import dataclass


class TokenType(Enum):
    """Token types for LaTeX parsing."""
    COMMAND = "command"        # \frac, \sqrt, etc.
    TEXT = "text"             # letters, numbers
    LBRACE = "lbrace"         # {
    RBRACE = "rbrace"         # }
    LBRACKET = "lbracket"     # [
    RBRACKET = "rbracket"     # ]
    LPAREN = "lparen"         # (
    RPAREN = "rparen"         # )
    CARET = "caret"           # ^
    UNDERSCORE = "underscore"  # _
    EQUALS = "equals"         # =
    PLUS = "plus"             # +
    MINUS = "minus"           # -
    MULTIPLY = "multiply"     # *
    EOF = "eof"               # end of input


@dataclass
class Token:
    """A token in the LaTeX expression."""
    type: TokenType
    value: str
    position: int


class Tokenizer:
    """Tokenizes LaTeX expressions into tokens."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)

    def current_char(self) -> Optional[str]:
        """Get the current character or None if at end."""
        if self.pos >= self.length:
            return None
        return self.text[self.pos]

    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Peek at character at current position + offset."""
        peek_pos = self.pos + offset
        if peek_pos >= self.length:
            return None
        return self.text[peek_pos]

    def advance(self) -> None:
        """Move to the next character."""
        self.pos += 1

    def skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        char = self.current_char()
        while char and char.isspace():
            self.advance()
            char = self.current_char()

    def read_command(self) -> str:
        """Read a LaTeX command starting with backslash."""
        command = ""
        if self.current_char() == '\\':
            command += '\\'
            self.advance()

            # Handle special single-character commands
            char = self.current_char()
            if char in ',;:>! ':
                command += char
                self.advance()
                return command

            # Read alphabetic characters
            while char and char.isalpha():
                command += char
                self.advance()
                char = self.current_char()

        return command

    def read_text(self) -> str:
        """Read consecutive text characters (letters, digits)."""
        text = ""
        char = self.current_char()
        while (char and
               (char.isalnum() or char in ".,;:!?")):
            text += char
            self.advance()
            char = self.current_char()
        return text

    def tokenize(self) -> List[Token]:
        """Tokenize the LaTeX expression."""
        tokens = []

        while self.pos < self.length:
            self.skip_whitespace()

            if self.pos >= self.length:
                break

            char = self.current_char()
            start_pos = self.pos

            if char == '\\':
                # LaTeX command
                command = self.read_command()

                # Special handling for \left and \right
                if command in ['\\left', '\\right'] and self.current_char():
                    bracket_char = self.current_char()
                    if bracket_char in '()[]{}':
                        self.advance()  # Consume the bracket
                        # Create a simple bracket token instead
                        tokens.append(Token(TokenType.TEXT, bracket_char, start_pos))
                    else:
                        tokens.append(Token(TokenType.COMMAND, command, start_pos))
                else:
                    tokens.append(Token(TokenType.COMMAND, command, start_pos))

            elif char == '{':
                tokens.append(Token(TokenType.LBRACE, char, start_pos))
                self.advance()

            elif char == '}':
                tokens.append(Token(TokenType.RBRACE, char, start_pos))
                self.advance()

            elif char == '[':
                tokens.append(Token(TokenType.LBRACKET, char, start_pos))
                self.advance()

            elif char == ']':
                tokens.append(Token(TokenType.RBRACKET, char, start_pos))
                self.advance()

            elif char == '(':
                tokens.append(Token(TokenType.LPAREN, char, start_pos))
                self.advance()

            elif char == ')':
                tokens.append(Token(TokenType.RPAREN, char, start_pos))
                self.advance()

            elif char == '^':
                tokens.append(Token(TokenType.CARET, char, start_pos))
                self.advance()

            elif char == '_':
                tokens.append(Token(TokenType.UNDERSCORE, char, start_pos))
                self.advance()

            elif char == '=':
                tokens.append(Token(TokenType.EQUALS, char, start_pos))
                self.advance()

            elif char == '+':
                tokens.append(Token(TokenType.PLUS, char, start_pos))
                self.advance()

            elif char == '-':
                tokens.append(Token(TokenType.MINUS, char, start_pos))
                self.advance()

            elif char == '*':
                tokens.append(Token(TokenType.MULTIPLY, char, start_pos))
                self.advance()

            elif char and (char.isalnum() or char in ".,;:!?"):
                # Text content
                text = self.read_text()
                tokens.append(Token(TokenType.TEXT, text, start_pos))

            else:
                # Treat unknown characters as text for now
                if char:
                    tokens.append(Token(TokenType.TEXT, char, start_pos))
                self.advance()

        tokens.append(Token(TokenType.EOF, "", self.pos))
        return tokens


class NodeType(Enum):
    """Types of nodes in the parse tree."""
    ROOT = "root"
    COMMAND = "command"
    GROUP = "group"
    TEXT = "text"
    OPERATOR = "operator"


@dataclass
class ParseNode:
    """A node in the parse tree."""
    type: NodeType
    value: str
    children: List['ParseNode']
    parent: Optional['ParseNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def add_child(self, child: 'ParseNode') -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)

    def __repr__(self) -> str:
        return f"ParseNode({self.type.value}, '{self.value}', {len(self.children)} children)"


class Parser:
    """Parses tokens into a tree structure."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.length = len(tokens)

    def current_token(self) -> Token:
        """Get the current token."""
        if self.pos >= self.length:
            return self.tokens[-1]  # EOF token
        return self.tokens[self.pos]

    def peek_token(self, offset: int = 1) -> Token:
        """Peek at token at current position + offset."""
        peek_pos = self.pos + offset
        if peek_pos >= self.length:
            return self.tokens[-1]  # EOF token
        return self.tokens[peek_pos]

    def advance(self) -> None:
        """Move to the next token."""
        if self.pos < self.length - 1:
            self.pos += 1

    def parse_group(self) -> ParseNode:
        """Parse a group enclosed in braces."""
        if self.current_token().type != TokenType.LBRACE:
            raise ValueError(f"Expected '{{' at position {self.current_token().position}")

        self.advance()  # Skip opening brace
        group_node = ParseNode(NodeType.GROUP, "", [])

        while (self.current_token().type != TokenType.RBRACE and
               self.current_token().type != TokenType.EOF):
            child = self.parse_expression()
            if child:
                group_node.add_child(child)

        if self.current_token().type != TokenType.RBRACE:
            raise ValueError(f"Missing closing brace '}}' at position {self.current_token().position}")

        self.advance()  # Skip closing brace
        return group_node

    def parse_command(self) -> ParseNode:
        """Parse a LaTeX command."""
        token = self.current_token()
        if token.type != TokenType.COMMAND:
            raise ValueError(f"Expected command at position {token.position}")

        command_node = ParseNode(NodeType.COMMAND, token.value, [])
        self.advance()

        # Handle commands that expect arguments
        if token.value in ['\\frac', '\\binom']:
            # These commands expect two arguments
            for _ in range(2):
                if self.current_token().type == TokenType.LBRACE:
                    arg = self.parse_group()
                    command_node.add_child(arg)
                elif self.current_token().type == TokenType.TEXT:
                    # Single character argument - wrap in group
                    text_token = self.current_token()
                    self.advance()
                    # For single character, create individual text nodes
                    if len(text_token.value) == 1:
                        text_node = ParseNode(NodeType.TEXT, text_token.value, [])
                        group = ParseNode(NodeType.GROUP, "", [text_node])
                        command_node.add_child(group)
                    else:
                        # Multiple characters - split into individual arguments
                        for char in text_token.value:
                            text_node = ParseNode(NodeType.TEXT, char, [])
                            group = ParseNode(NodeType.GROUP, "", [text_node])
                            command_node.add_child(group)
                        break  # We've consumed all characters
                elif self.current_token().type not in [TokenType.EOF, TokenType.RBRACE]:
                    # Other single token argument - wrap in group
                    single_arg = self.parse_primary()
                    if single_arg:
                        group = ParseNode(NodeType.GROUP, "", [single_arg])
                        command_node.add_child(group)

        elif token.value in ['\\sqrt']:
            # Handle optional argument [n] for \sqrt
            optional_present = False
            if self.current_token().type == TokenType.LBRACKET:
                optional_present = True
                self.advance()  # Skip [
                optional_arg = ParseNode(NodeType.GROUP, "", [])
                while (self.current_token().type != TokenType.RBRACKET and
                       self.current_token().type != TokenType.EOF):
                    child = self.parse_expression()
                    if child:
                        optional_arg.add_child(child)

                if self.current_token().type == TokenType.RBRACKET:
                    self.advance()  # Skip ]
                    # Mark this as an optional argument (we'll handle it in LaTeX generation)
                    optional_arg.value = "optional"
                    command_node.add_child(optional_arg)

            # Main argument
            if self.current_token().type == TokenType.LBRACE:
                arg = self.parse_group()
                command_node.add_child(arg)
            elif self.current_token().type == TokenType.TEXT:
                # Single character argument
                text_token = self.current_token()
                self.advance()
                text_node = ParseNode(NodeType.TEXT, text_token.value, [])
                group = ParseNode(NodeType.GROUP, "", [text_node])
                command_node.add_child(group)
            elif self.current_token().type not in [TokenType.EOF, TokenType.RBRACE]:
                single_arg = self.parse_primary()
                if single_arg:
                    group = ParseNode(NodeType.GROUP, "", [single_arg])
                    command_node.add_child(group)

        return command_node

    def parse_primary(self) -> Optional[ParseNode]:
        """Parse a primary expression (text, command, or group)."""
        token = self.current_token()

        if token.type == TokenType.EOF:
            return None

        elif token.type == TokenType.COMMAND:
            return self.parse_command()

        elif token.type == TokenType.LBRACE:
            return self.parse_group()

        elif token.type == TokenType.TEXT:
            node = ParseNode(NodeType.TEXT, token.value, [])
            self.advance()
            return node

        elif token.type in [TokenType.EQUALS, TokenType.PLUS, TokenType.MINUS,
                           TokenType.MULTIPLY, TokenType.LPAREN, TokenType.RPAREN]:
            node = ParseNode(NodeType.OPERATOR, token.value, [])
            self.advance()
            return node

        else:
            # Skip unexpected tokens
            self.advance()
            return None

    def parse_expression(self) -> Optional[ParseNode]:
        """Parse an expression with operators like ^ and _."""
        left = self.parse_primary()
        if not left:
            return None

        # Handle superscript (^) and subscript (_)
        while self.current_token().type in [TokenType.CARET, TokenType.UNDERSCORE]:
            op_token = self.current_token()
            self.advance()

            # Create operator node
            op_node = ParseNode(NodeType.OPERATOR, op_token.value, [])
            op_node.add_child(left)

            # Parse right side
            if self.current_token().type == TokenType.LBRACE:
                right = self.parse_group()
            elif self.current_token().type not in [TokenType.EOF, TokenType.RBRACE]:
                right = self.parse_primary()
                if right:
                    # Wrap single token in group for consistency
                    group = ParseNode(NodeType.GROUP, "", [right])
                    right = group
            else:
                right = None

            if right:
                op_node.add_child(right)

            left = op_node

        return left

    def parse(self) -> ParseNode:
        """Parse the token list into a tree."""
        root = ParseNode(NodeType.ROOT, "", [])

        while self.current_token().type != TokenType.EOF:
            expr = self.parse_expression()
            if expr:
                root.add_child(expr)

        return root


class Normalizer:
    """Applies normalization rules to the parse tree."""

    def __init__(self):
        # Font/styling commands to remove
        self.font_commands = {
            '\\mathbf', '\\mathrm', '\\mathit', '\\mathcal', '\\mathbb',
            '\\boldsymbol', '\\textbf', '\\scriptstyle', '\\textstyle', '\\mbox'
        }

        # Accent commands to remove
        self.accent_commands = {
            '\\vec', '\\dot', '\\ddot', '\\bar', '\\tilde', '\\hat',
            '\\overline', '\\underline', '\\widehat', '\\widetilde'
        }

        # Spacing commands to remove
        self.spacing_commands = {
            '\\,', '\\;', '\\:', '\\>', '\\!', '\\ ', '\\quad', '\\qquad'
        }

        # Command synonyms to replace
        self.command_synonyms = {
            '\\over': '\\frac'
        }

        # Commands that should be replaced with brackets only
        self.bracket_commands = {
            '\\left', '\\right'
        }

    def normalize(self, node: ParseNode) -> ParseNode:
        """Apply normalization rules to the parse tree."""
        return self._normalize_node(node)

    def _normalize_node(self, node: ParseNode) -> ParseNode:
        """Recursively normalize a node and its children."""
        if node.type == NodeType.COMMAND:
            return self._normalize_command(node)
        elif node.type == NodeType.GROUP:
            return self._normalize_group(node)
        elif node.type == NodeType.ROOT:
            return self._normalize_root(node)
        else:
            # For text and operator nodes, just normalize children if any
            normalized_children = []
            for child in node.children:
                normalized_child = self._normalize_node(child)
                if normalized_child:
                    normalized_children.append(normalized_child)

            new_node = ParseNode(node.type, node.value, normalized_children)
            return new_node

    def _normalize_command(self, node: ParseNode) -> Optional[ParseNode]:
        """Normalize a command node."""
        command = node.value

        # Remove font commands - return only their content
        if command in self.font_commands:
            if len(node.children) == 1:
                # Return the content of the font command
                return self._normalize_node(node.children[0])
            else:
                # Multiple children - create a group
                normalized_children = []
                for child in node.children:
                    normalized_child = self._normalize_node(child)
                    if normalized_child:
                        normalized_children.append(normalized_child)
                return ParseNode(NodeType.GROUP, "", normalized_children)

        # Remove accent commands - return only their content
        if command in self.accent_commands:
            if len(node.children) == 1:
                return self._normalize_node(node.children[0])
            else:
                normalized_children = []
                for child in node.children:
                    normalized_child = self._normalize_node(child)
                    if normalized_child:
                        normalized_children.append(normalized_child)
                return ParseNode(NodeType.GROUP, "", normalized_children)

        # Remove spacing commands entirely
        if command in self.spacing_commands:
            return None  # Remove the command

        # Replace command synonyms
        if command in self.command_synonyms:
            new_command = self.command_synonyms[command]
            normalized_children = []
            for child in node.children:
                normalized_child = self._normalize_node(child)
                if normalized_child:
                    normalized_children.append(normalized_child)
            return ParseNode(NodeType.COMMAND, new_command, normalized_children)

        # Handle \left and \right - replace with plain brackets
        if command in ['\\left', '\\right']:
            # These commands should be followed by a bracket character in the text
            # We need to look at what comes next in the tokenization
            # For now, return None and let the next token (the bracket) be processed normally
            return None

        # Regular command - normalize children
        normalized_children = []
        for child in node.children:
            normalized_child = self._normalize_node(child)
            if normalized_child:
                normalized_children.append(normalized_child)

        return ParseNode(NodeType.COMMAND, command, normalized_children)

    def _normalize_group(self, node: ParseNode) -> ParseNode:
        """Normalize a group node."""
        normalized_children = []
        for child in node.children:
            normalized_child = self._normalize_node(child)
            if normalized_child:
                normalized_children.append(normalized_child)

        # Check if this group can be simplified
        # Simplify groups with a single text node, unless it's part of a command argument
        if (len(normalized_children) == 1 and
            normalized_children[0].type == NodeType.TEXT and
            node.parent and node.parent.type != NodeType.COMMAND):
            # Return the text node directly (remove unnecessary braces)
            return normalized_children[0]

        return ParseNode(NodeType.GROUP, node.value, normalized_children)

    def _normalize_root(self, node: ParseNode) -> ParseNode:
        """Normalize the root node."""
        normalized_children = []
        i = 0
        children = node.children

        while i < len(children):
            child = children[i]

            # Check for \over pattern: {a} \over {b}
            if (child.type == NodeType.COMMAND and child.value == '\\over' and
                i > 0 and i < len(children) - 1):

                # Look for preceding group
                prev_child = children[i - 1]
                next_child = children[i + 1]

                if (prev_child.type == NodeType.GROUP and
                    next_child.type == NodeType.GROUP):

                    # Remove the previous child (we'll replace it)
                    if normalized_children:
                        normalized_children.pop()

                    # Create \frac{a}{b} - ensure both arguments stay as groups
                    frac_node = ParseNode(NodeType.COMMAND, '\\frac', [])

                    # Normalize the children but keep them as groups
                    norm_prev = self._normalize_node(prev_child)
                    norm_next = self._normalize_node(next_child)

                    # If normalization simplified a group to text, wrap it back in a group
                    if norm_prev.type != NodeType.GROUP:
                        norm_prev = ParseNode(NodeType.GROUP, "", [norm_prev])
                    if norm_next.type != NodeType.GROUP:
                        norm_next = ParseNode(NodeType.GROUP, "", [norm_next])

                    frac_node.add_child(norm_prev)
                    frac_node.add_child(norm_next)

                    normalized_children.append(frac_node)
                    i += 2  # Skip the next child as we've consumed it
                    continue

            normalized_child = self._normalize_node(child)
            if normalized_child:
                normalized_children.append(normalized_child)
            i += 1

        return ParseNode(NodeType.ROOT, "", normalized_children)


class LaTeXGenerator:
    """Converts a parse tree back to LaTeX string."""

    def generate(self, node: ParseNode) -> str:
        """Generate LaTeX string from parse tree."""
        if node.type == NodeType.ROOT:
            parts = []
            for child in node.children:
                part = self.generate(child)
                if part:
                    parts.append(part)

            # Smart spacing - add spaces around operators but not around commands
            result = ""
            for i, part in enumerate(parts):
                if i > 0:
                    prev_part = parts[i-1]
                    # Add space if both parts are "words" (not commands starting with \)
                    if (not part.startswith('\\') and not prev_part.startswith('\\') and
                        not part in '(){}[]' and not prev_part in '(){}[]' and
                        not part in '+-=^_' and not prev_part in '+-=^_'):
                        result += " "
                result += part
            return result

        elif node.type == NodeType.COMMAND:
            result = node.value

            # Special handling for \sqrt with optional argument
            if node.value == "\\sqrt" and len(node.children) >= 1:
                first_child = node.children[0]
                if first_child.type == NodeType.GROUP and first_child.value == "optional":
                    # This is an optional argument - use brackets
                    result += "[" + self.generate(first_child) + "]"
                    # Add remaining children as regular arguments
                    for child in node.children[1:]:
                        if child.type == NodeType.GROUP:
                            result += "{" + self.generate(child) + "}"
                        else:
                            result += self.generate(child)
                else:
                    # No optional argument - treat all as regular arguments
                    for child in node.children:
                        if child.type == NodeType.GROUP:
                            result += "{" + self.generate(child) + "}"
                        else:
                            result += self.generate(child)
            else:
                # Regular command handling
                for child in node.children:
                    if child.type == NodeType.GROUP:
                        result += "{" + self.generate(child) + "}"
                    else:
                        result += self.generate(child)
            return result

        elif node.type == NodeType.GROUP:
            return "".join(self.generate(child) for child in node.children)

        elif node.type == NodeType.TEXT:
            return node.value

        elif node.type == NodeType.OPERATOR:
            if node.value in ['^', '_']:
                # Handle superscript/subscript
                if len(node.children) >= 2:
                    base = self.generate(node.children[0])
                    exp = self.generate(node.children[1])
                    return f"{base}{node.value}{{{exp}}}"
                else:
                    return node.value
            else:
                return node.value

        return ""


def normalize_latex(latex_str: str) -> str:
    """
    Normalize a LaTeX expression.

    Args:
        latex_str: Input LaTeX expression

    Returns:
        Normalized LaTeX string

    Raises:
        ValueError: For malformed LaTeX
    """
    try:
        # Tokenize
        tokenizer = Tokenizer(latex_str)
        tokens = tokenizer.tokenize()

        # Parse
        parser = Parser(tokens)
        tree = parser.parse()

        # Normalize (Phase 2)
        normalizer = Normalizer()
        normalized_tree = normalizer.normalize(tree)

        # Generate LaTeX
        generator = LaTeXGenerator()
        result = generator.generate(normalized_tree)

        return result.strip()

    except Exception as e:
        raise ValueError(f"Error parsing LaTeX: {e}")


# Basic validation function
def validate_latex(latex_str: str) -> bool:
    """
    Basic validation of LaTeX syntax.

    Args:
        latex_str: LaTeX expression to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        normalize_latex(latex_str)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    # Basic test
    test_expressions = [
        "x^2",
        "\\frac{a}{b}",
        "\\sqrt{x}",
        "x^{2}+y",
        "\\frac{x+1}{y-2}"
    ]

    print("Testing LaTeX Normalizer (Phase 1):")
    for expr in test_expressions:
        try:
            result = normalize_latex(expr)
            print(f"Input:  {expr}")
            print(f"Output: {result}")
            print()
        except Exception as e:
            print(f"Error with '{expr}': {e}")
            print()