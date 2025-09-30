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

ACCENT_COMMANDS = set(['\\vec', '\\dot', '\\ddot', '\\bar', '\\tilde', '\\hat',
                       '\\overline', '\\widehat', '\\widetilde'])
COMMAND_SINGLE_ARGUMENT = ACCENT_COMMANDS

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

            # Handle special single-character commands including escaped braces
            char = self.current_char()
            if char in ',;:>! {}':
                command += char
                self.advance()
                return command

            # Read alphabetic characters
            while char and char.isalpha():
                command += char
                self.advance()
                char = self.current_char()

        return command

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

            elif char and (char.isalnum() or char in ".,;:!?=+-*"):
                # Text content
                tokens.append(Token(TokenType.TEXT, char, start_pos))
                self.advance()

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
        self.brace_stack = []  # Track brace nesting for validation

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
            raise LaTeXError(f"Expected '{{' at position {self.current_token().position}")

        brace_pos = self.current_token().position
        self.brace_stack.append(brace_pos)
        self.advance()  # Skip opening brace
        group_node = ParseNode(NodeType.GROUP, "", [])

        while (self.current_token().type != TokenType.RBRACE and
               self.current_token().type != TokenType.EOF):
            child = self.parse_expression()
            if child:
                group_node.add_child(child)

        if self.current_token().type != TokenType.RBRACE:
            raise LaTeXError(f"Unmatched opening brace '{{' at position {brace_pos}")

        self.brace_stack.pop()
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
        if token.value in ['\\frac','\\binom']:
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
        elif token.value in COMMAND_SINGLE_ARGUMENT:
            if self.current_token().type == TokenType.LBRACE:
                arg = self.parse_group()
                command_node.add_child(arg)
            else:
                # Single character argument
                next_node = self.parse_primary()
                if next_node:
                    group = ParseNode(NodeType.GROUP, "", [next_node])
                    command_node.add_child(group)
                else:
                    raise LaTeXError(f"Expected argument for command {token.value} at position {token.position}")    

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

        elif token.type in [TokenType.LPAREN, TokenType.RPAREN,
                           TokenType.LBRACKET, TokenType.RBRACKET]:
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
            elif self.current_token().type == TokenType.TEXT:
                # Special handling for text tokens after ^ or _
                text_token = self.current_token()
                self.advance()
                
                # Check if we need to split this token
                split_pos = self._find_subscript_split_position(text_token.value)
                
                if split_pos < len(text_token.value):
                    # Split the token
                    subscript_part = text_token.value[:split_pos]
                    remaining_part = text_token.value[split_pos:]
                    
                    # Create subscript/superscript with the first part
                    text_node = ParseNode(NodeType.TEXT, subscript_part, [])
                    group = ParseNode(NodeType.GROUP, "", [text_node])
                    right = group
                    
                    # Put back the remaining characters as a new token
                    if remaining_part:
                        self.pos -= 1  # Go back one position
                        # Modify the current token to have the remaining text
                        self.tokens[self.pos] = Token(TokenType.TEXT, remaining_part, text_token.position + split_pos)
                else:
                    # Use the entire token
                    text_node = ParseNode(NodeType.TEXT, text_token.value, [])
                    group = ParseNode(NodeType.GROUP, "", [text_node])
                    right = group
                    
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
    
    def _find_subscript_split_position(self, text: str) -> int:
        """Find where to split a text token for subscript/superscript parsing.
        
        For unbraced subscripts/superscripts, only take the first character.
        This matches standard LaTeX behavior where a_12 means a_{1}2, not a_{12}.
        """
        if not text:
            return 0
            
        # For unbraced subscripts/superscripts, always take only the first character
        return 1  # Default: single character

    def parse(self) -> ParseNode:
        """Parse the token list into a tree."""
        root = ParseNode(NodeType.ROOT, "", [])

        while self.current_token().type != TokenType.EOF:
            expr = self.parse_expression()
            if expr:
                root.add_child(expr)

        # Check for unmatched braces
        if self.brace_stack:
            raise LaTeXError(f"Unmatched opening brace at position {self.brace_stack[0]}")

        return root


class LaTeXError(ValueError):
    """Custom exception for LaTeX parsing errors."""
    pass


class Normalizer:
    """Applies normalization rules to the parse tree."""

    def __init__(self):
        # Font/styling commands to remove
        self.font_commands = {
            '\\mathbf', '\\mathrm', '\\mathit', '\\mathcal', '\\mathbb',
            '\\boldsymbol', '\\textbf', '\\scriptstyle', '\\textstyle', '\\mbox',
            '\\operatorname', '\\text', '\\rm', '\\bf', '\\it', '\\displaystyle'
        }

        # Accent commands to remove
        # self.accent_commands = {
        #     '\\vec', '\\dot', '\\ddot', '\\bar', '\\tilde', '\\hat',
        #     '\\overline', '\\underline', '\\widehat', '\\widetilde'
        # }

        # Spacing commands to remove
        self.spacing_commands = {
            '\\,', '\\;', '\\:', '\\>', '\\!', '\\ ', '\\quad', '\\qquad'
        }

        # Command synonyms to replace
        self.command_synonyms = {
            '\\le': '\\leq',
            '\\ge': '\\geq'
        }

        # Commands that should be replaced with brackets only
        self.bracket_commands = {
            '\\left', '\\right'
        }

        # Unsupported constructs that should cause errors
        self.unsupported_commands = {
            '\\limits', '\\nolimits', '\\begin', '\\end', '\\binom'
        }

        # Matrix environments (unsupported)
        self.matrix_environments = {
            'matrix', 'pmatrix', 'bmatrix', 'Bmatrix', 'vmatrix', 'Vmatrix', 'array'
        }

    def normalize(self, node: ParseNode) -> ParseNode:
        """Apply normalization rules to the parse tree."""
        # First validate the tree for errors
        self._validate_tree(node)

        result = self._normalize_node(node)
        if result is None:
            # If root gets normalized to None, return empty root
            return ParseNode(NodeType.ROOT, "", [])
        return result

    def _validate_tree(self, node: ParseNode) -> None:
        """Validate the parse tree for errors and unsupported constructs."""
        if node.type == NodeType.COMMAND:
            self._validate_command(node)
        elif node.type == NodeType.GROUP:
            self._validate_group(node)
        # Recursively validate children
        for child in node.children:
            self._validate_tree(child)

    def _validate_command(self, node: ParseNode) -> None:
        """Validate a command node."""
        command = node.value

        # Check for unsupported commands
        if command in self.unsupported_commands:
            raise LaTeXError(f"Unsupported command: {command}")

        # Check for matrix environments
        if command == '\\begin' and len(node.children) > 0:
            first_child = node.children[0]
            if (first_child.type == NodeType.GROUP and
                len(first_child.children) > 0 and
                first_child.children[0].type == NodeType.TEXT):
                env_name = first_child.children[0].value
                if env_name in self.matrix_environments:
                    raise LaTeXError(f"Matrix environments are not supported: \\begin{{{env_name}}}")

        # Validate command arguments
        if command in ['\\frac']:
            if len(node.children) != 2:
                raise LaTeXError(f"Command {command} requires exactly 2 arguments, got {len(node.children)}")
            # Check for empty arguments
            for i, child in enumerate(node.children):
                if self._is_empty_group(child):
                    raise LaTeXError(f"Command {command} has empty argument {i+1}")

        elif command == '\\sqrt':
            if len(node.children) == 0:
                raise LaTeXError("Command \\sqrt requires at least 1 argument")
            # Check for empty main argument (last one)
            main_arg = node.children[-1]
            if self._is_empty_group(main_arg):
                raise LaTeXError("Command \\sqrt has empty main argument")
            

    def _validate_group(self, node: ParseNode) -> ParseNode | None:
        """Validate a group node."""
        pass

    def _check_group_commands(self, node: ParseNode) -> ParseNode | None:
        # some commands operate on the group in which they are contained
        # e.g. {a \over b} -> \frac{a}{b}
        group_commands = {'\\over', '\\atop', '\\choose'}
        if node.children:
            # Check if the group contains any of the group commands
            for child in node.children:
                if child.type == NodeType.COMMAND and child.value in group_commands:
                    return self._transform_group_command(node, child.value)
        return None

    def _transform_group_command(self, group_node: ParseNode, command: str) -> ParseNode:
        """Transform a group containing a group command into a structured command node."""
        if command == '\\over':
            command_name = '\\frac'
        elif command == '\\choose':
            command_name = '\\binom'
        else:
            raise LaTeXError(f"Unknown group command: {command}")

        # Split the group children at the command
        # find the index of the group command
        index = next((i for i, child in enumerate(group_node.children)
                       if child.type == NodeType.COMMAND and child.value == command), None)
        if index is None:
            raise LaTeXError(f"Group command {command} not found")
        # Split into 2 parts
        parts = [group_node.children[:index], group_node.children[index + 1:]]

        # Create the new command node
        command_node = ParseNode(NodeType.COMMAND, command_name, [])

        # Create groups for each part
        for part in parts:
            if len(part) == 1 and part[0].type == NodeType.GROUP:
                # If the part is already a group, use it directly
                arg_group = part[0]
            else:
                arg_group = ParseNode(NodeType.GROUP, "", part)
            command_node.add_child(arg_group)

        return command_node         

    def _is_empty_group(self, node: ParseNode) -> bool:
        """Check if a group is empty or contains only whitespace."""
        if node.type != NodeType.GROUP:
            return False

        if len(node.children) == 0:
            return True

        # Check if all children are empty text nodes
        for child in node.children:
            if child.type == NodeType.TEXT and child.value.strip():
                return False
            elif child.type != NodeType.TEXT:
                return False

        return True

    def _normalize_node(self, node: ParseNode) -> Optional[ParseNode]:
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
        # if command in self.accent_commands:
        #     if len(node.children) == 1:
        #         return self._normalize_node(node.children[0])
        #     else:
        #         normalized_children = []
        #         for child in node.children:
        #             normalized_child = self._normalize_node(child)
        #             if normalized_child:
        #                 normalized_children.append(normalized_child)
        #         return ParseNode(NodeType.GROUP, "", normalized_children)

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

        # Handle \left and \right - remove the commands but preserve following content
        if command in ['\\left', '\\right']:
            # Return None - the command will be removed, and following brackets 
            # will be processed as regular tokens
            return None

        # Convert \neq to \ne for standard form
        if command == '\\neq':
            return ParseNode(NodeType.COMMAND, '\\ne', [])

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
                # Don't add empty groups
                if (normalized_child.type == NodeType.GROUP and 
                    len(normalized_child.children) == 0):
                    continue
                normalized_children.append(normalized_child)

        res = ParseNode(NodeType.GROUP, node.value, normalized_children)
        # Check for group commands like \over, \atop, \choose
        transformed_node = self._check_group_commands(res)
        if transformed_node:
            res = transformed_node
        
        return res

    def _normalize_root(self, node: ParseNode) -> ParseNode:
        """Normalize the root node."""
        normalized_children = []
        i = 0
        children = node.children

        while i < len(children):
            child = children[i]
            normalized_child = self._normalize_node(child)
            if normalized_child:
                normalized_children.append(normalized_child)
            i += 1

        res = ParseNode(NodeType.ROOT, "", normalized_children)

        # Check for group commands like \over, \atop, \choose
        transformed_node = self._check_group_commands(res)
        if transformed_node:
            res = transformed_node

        return res


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

            # Simple spacing - add spaces between most elements
            result = ""
            for i, part in enumerate(parts):
                if i > 0:
                    prev_part = parts[i-1]
                    
                    # Always add space around operators
                    if part in ['=', '\\neq', '\\ne', '\\leq', '\\geq', '\\lt', '\\gt', '<', '>', '\\pm', '\\mp']:
                        result += " "
                    elif prev_part in ['=', '\\neq', '\\ne', '\\leq', '\\geq', '\\lt', '\\gt', '<', '>', '\\pm', '\\mp']:
                        result += " "
                    # Add space between commands and text/other elements
                    elif prev_part.startswith('\\') and not part.startswith('\\') and not part in '(){}[]^_+-=':
                        result += " "
                    elif not prev_part.startswith('\\') and part.startswith('\\') and not prev_part in '(){}[]^_+-=':
                        result += " "
                    # Add space between text elements
                    elif (not part.startswith('\\') and not prev_part.startswith('\\') and
                          not part in '(){}[]^_+-=' and not prev_part in '(){}[]^_+-='):
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
            # Generate content for group children with liberal spacing
            parts = []
            for child in node.children:
                part = self.generate(child)
                if part:
                    parts.append(part)
            
            # Add spaces between elements within groups too
            result = ""
            for i, part in enumerate(parts):
                if i > 0:
                    prev_part = parts[i-1]
                    
                    # Add space around operators
                    if part in ['=', '\\neq', '\\ne', '\\leq', '\\geq', '\\lt', '\\gt', '<', '>']:
                        result += " "
                    elif prev_part in ['=', '\\neq', '\\ne', '\\leq', '\\geq', '\\lt', '\\gt', '<', '>']:
                        result += " "
                    # Add space between commands and text
                    elif prev_part.startswith('\\') and not part.startswith('\\') and not part in '(){}[]^_+-=':
                        result += " "
                    elif not prev_part.startswith('\\') and part.startswith('\\') and not prev_part in '(){}[]^_+-=':
                        result += " "
                
                result += part
            return result + ' '

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
        LaTeXError: For malformed LaTeX or unsupported constructs
        ValueError: For other parsing errors
    """
    if not latex_str or not latex_str.strip():
        raise LaTeXError("Input LaTeX expression is empty")

    try:
        # Tokenize
        tokenizer = Tokenizer(latex_str)
        tokens = tokenizer.tokenize()

        # Parse
        parser = Parser(tokens)
        tree = parser.parse()

        # Normalize (includes validation)
        normalizer = Normalizer()
        normalized_tree = normalizer.normalize(tree)

        # Generate LaTeX
        generator = LaTeXGenerator()
        result = generator.generate(normalized_tree)
        return result.strip()

    except LaTeXError:
        # Re-raise LaTeX-specific errors as-is
        raise
    except Exception as e:
        # Wrap other errors with more context
        raise LaTeXError(f"Error processing LaTeX expression '{latex_str}': {e}")


# Validation function
def validate_latex(latex_str: str) -> bool:
    """
    Validate LaTeX syntax without normalizing.

    Args:
        latex_str: LaTeX expression to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        normalize_latex(latex_str)
        return True
    except (LaTeXError, ValueError):
        return False


def get_latex_errors(latex_str: str) -> List[str]:
    """
    Get detailed error messages for invalid LaTeX.

    Args:
        latex_str: LaTeX expression to check

    Returns:
        List of error messages (empty if valid)
    """
    try:
        normalize_latex(latex_str)
        return []
    except (LaTeXError, ValueError) as e:
        return [str(e)]


def run_test_suite():
    """Run the basic test suite for the LaTeX normalizer."""

    # Phase 4: Basic test cases covering all requirements
    test_cases = [
        # Requirements examples - normalization
        ("{x+1}\\over{4}", "\\frac{x+1}{4}", "Command synonym: \\over → \\frac"),
        ("{n}", "n", "Remove unnecessary brackets"),
        ("\\frac{{x+2}n}{2x}", "\\frac{x+2n}{2x}", "Simplify nested brackets"),
        ("\\frac12", "\\frac{1}{2}", "Add missing brackets for commands"),
        ("\\sqrt[n]3", "\\sqrt[n]{3}", "Add missing brackets for optional args"),
        ("\\mathbf{1+x}", "1+x", "Remove font styling"),
        ("\\frac{\\scriptstyle{WF}}{2}", "\\frac{WF}{2}", "Remove nested styling"),

        # Additional normalization
        ("x^2", "x^{2}", "Normalize exponents"),
        ("\\left(x+1\\right)", "(x+1)", "Remove \\left/\\right"),
        ("\\vec{x}+y", "x+y", "Remove accents"),
        ("x\\,+\\;y", "x+y", "Remove spacing"),
        ("\\mbox{hello}", "hello", "Remove text styling"),

        # Edge cases
        ("\\sqrt{x}", "\\sqrt{x}", "Valid sqrt unchanged"),
        ("\\frac{\\text{top}}{\\text{bottom}}", "\\frac{top}{bottom}", "Nested text removal"),
        ("a_n", "a_{n}", "Normalize subscripts"),
        ("\\sqrt[3]{x^2}", "\\sqrt[3]{x^{2}}", "Complex nested expression"),
        ("\\rm{x}\\bf{y}", "x y", "Multiple font commands"),  # Updated: more liberal spacing
        ("x\\quad y\\quad z", "x y z", "Multiple spacing commands"),  # Updated: more liberal spacing

        # Fixed problematic cases
        ("\\left [ { e } ^ { \\mbox { m } } \\right ]", "[e^{m}]", "Left/right brackets with mbox"),
        ("\\sum _ { { u \\geq x } } { { \\mbox { X } - \\mbox { p } } }", "\\sum_{u \\geq x} X-p", "Sum with subscript and mbox expressions"),  # Updated: space before X
        ("{{\\sum{\\mbox{x}}-\\frac{\\beta}{k}}\\neq\\frac{{T-\\mbox{o}}}{{y}^{e}\\left(\\phi\\right)}}", "\\sum x-\\frac{\\beta}{k} \\ne \\frac{T-o}{y^{e}(\\phi)}", "Complex expression with nested braces and commands"),
        
        # Subscript normalization - UPDATED: only first character in unbraced subscripts
        ("f(x) = a_nx^n + a_{n-1}x^{n-1}+\\cdots+a_1x + a_0", "f(x) = a_{n} x^{n}+a_{n-1} x^{n-1}+\\cdots+a_{1} x+a_{0}", "Single character variable subscripts"),
        ("a_12^n", "a_{1} 2^{n}", "Only first character in unbraced subscripts"),  # UPDATED
        ("a_1b^n", "a_{1} b^{n}", "Mixed digit-letter subscripts"),  # Updated: space after subscript
        ("y_a12^x", "y_{a} 12^{x}", "Letter followed by digits subscripts"),  # UPDATED: only first char
        ("z_12a^b", "z_{1} 2a^{b}", "Digits followed by letter subscripts"),  # UPDATED: only first char
        
        # Explicit braces preserve full subscripts
        ("a_{12}^n", "a_{12}^{n}", "Explicit braces preserve full subscripts"),
        
        # Command-text spacing cases
        ("\\tan \\mbox { h }", "\\tan h", "Command followed by single character in mbox"),
        ("\\mbox { l } \\left ( \\mbox { h } \\right )", "l(h)", "Mbox with left/right parentheses"),
        ("\\log 5", "\\log 5", "Command followed by single digit"),
        
        # Escaped braces
        ("\\{ T \\}", "\\{ T \\}", "Escaped braces preserved"),
    ]

    # Error cases
    error_cases = [
        ("\\frac{}{}", "Empty fraction arguments"),
        ("{x+1", "Unmatched braces"),
        ("\\limits", "Unsupported command"),
        ("", "Empty input"),
        ("\\begin{matrix}1\\end{matrix}", "Matrix environment"),
        ("\\sqrt{}", "Empty sqrt"),
        ("\\binom{n}{k}", "Unsupported binom"),
    ]

    print("=== LATEX NORMALIZER TEST SUITE ===")
    print(f"Testing {len(test_cases)} normalization cases...")

    passed = 0
    for input_expr, expected, description in test_cases:
        try:
            result = normalize_latex(input_expr)
            if result == expected:
                print(f"✓ {description}")
                passed += 1
            else:
                print(f"✗ {description}")
                print(f"  Expected: {expected}")
                print(f"  Got:      {result}")
        except Exception as e:
            print(f"✗ {description} - Error: {e}")

    print(f"\nTesting {len(error_cases)} error cases...")
    error_passed = 0
    for input_expr, description in error_cases:
        try:
            result = normalize_latex(input_expr)
            print(f"✗ {description} - Should have failed but got: {result}")
        except (LaTeXError, ValueError):
            print(f"✓ {description}")
            error_passed += 1
        except Exception as e:
            print(f"✗ {description} - Unexpected error: {e}")

    total_tests = len(test_cases) + len(error_cases)
    total_passed = passed + error_passed

    print(f"\n=== RESULTS ===")
    print(f"Normalization: {passed}/{len(test_cases)} passed")
    print(f"Error handling: {error_passed}/{len(error_cases)} passed")
    print(f"Overall: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED - LaTeX Normalizer is ready!")
        return True
    else:
        print(f"❌ {total_tests - total_passed} tests failed")
        return False


if __name__ == "__main__":
    result = normalize_latex('\\hat\\nu_i')
    print("Normalized LaTeX:", result)
    # run_test_suite()