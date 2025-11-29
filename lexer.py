"""
TinyGame Compiler - Lexical Analyzer
Tokenizes input source code into meaningful tokens
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

# ============================================================================
# TOKEN TYPES
# ============================================================================

class TokenType(Enum):
    """All token types in TinyGame language"""
    
    # Keywords
    PLAYER = "PLAYER"
    ENEMY = "ENEMY"
    MOVE = "MOVE"
    SET = "SET"
    IF = "IF"
    PRINT = "PRINT"
    END = "END"
    
    # Directions (for move command)
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    
    # Operators
    ASSIGN = "ASSIGN"          # =
    PLUS = "PLUS"              # +
    MINUS = "MINUS"            # -
    MULTIPLY = "MULTIPLY"      # *
    DIVIDE = "DIVIDE"          # /
    EQUAL = "EQUAL"            # ==
    NOT_EQUAL = "NOT_EQUAL"    # !=
    GREATER = "GREATER"        # >
    LESS = "LESS"              # 
    
    # Delimiters
    LBRACE = "LBRACE"          # {
    RBRACE = "RBRACE"          # }
    LPAREN = "LPAREN"          # (
    RPAREN = "RPAREN"          # )
    SEMICOLON = "SEMICOLON"    # ;
    DOT = "DOT"                # .
    COMMA = "COMMA"            # ,
    
    # Literals and Identifiers
    INTEGER = "INTEGER"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"
    
    # Special
    EOF = "EOF"
    NEWLINE = "NEWLINE"


# ============================================================================
# TOKEN DATA CLASS
# ============================================================================

@dataclass
class Token:
    """Represents a single token"""
    type: TokenType
    value: any
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type.name}, {repr(self.value)}, {self.line}:{self.column})"


# ============================================================================
# LEXER CLASS
# ============================================================================

class Lexer:
    """Lexical Analyzer for TinyGame"""
    
    # Keywords mapping
    KEYWORDS = {
        'player': TokenType.PLAYER,
        'enemy': TokenType.ENEMY,
        'move': TokenType.MOVE,
        'set': TokenType.SET,
        'if': TokenType.IF,
        'print': TokenType.PRINT,
        'end': TokenType.END,
        'up': TokenType.UP,
        'down': TokenType.DOWN,
        'left': TokenType.LEFT,
        'right': TokenType.RIGHT,
    }
    
    def __init__(self, source_code: str):
        """Initialize lexer with source code"""
        self.source = source_code
        self.pos = 0
        self.line = 1
        self.column = 1
        self.current_char = self.source[0] if source_code else None
        self.tokens = []
    
    def error(self, msg: str):
        """Raise lexical error"""
        raise Exception(f"Lexical Error at {self.line}:{self.column}: {msg}")
    
    def advance(self):
        """Move to next character"""
        if self.current_char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        
        self.pos += 1
        if self.pos >= len(self.source):
            self.current_char = None
        else:
            self.current_char = self.source[self.pos]
    
    def peek(self, offset: int = 1) -> Optional[str]:
        """Look ahead at next character(s) without consuming"""
        peek_pos = self.pos + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def skip_whitespace(self):
        """Skip whitespace characters"""
        while self.current_char and self.current_char in ' \t\r\n':
            self.advance()
    
    def skip_comment(self):
        """Skip single-line comments (// style)"""
        if self.current_char == '/' and self.peek() == '/':
            while self.current_char and self.current_char != '\n':
                self.advance()
            if self.current_char == '\n':
                self.advance()
    
    def read_integer(self) -> Token:
        """Read integer literal"""
        start_line = self.line
        start_col = self.column
        num_str = ''
        
        while self.current_char and self.current_char.isdigit():
            num_str += self.current_char
            self.advance()
        
        return Token(TokenType.INTEGER, int(num_str), start_line, start_col)
    
    def read_identifier(self) -> Token:
        """Read identifier or keyword"""
        start_line = self.line
        start_col = self.column
        id_str = ''
        
        # First character must be lowercase letter
        while self.current_char and (self.current_char.isalnum() and self.current_char.islower()):
            id_str += self.current_char
            self.advance()
        
        # Check if it's a keyword
        token_type = self.KEYWORDS.get(id_str, TokenType.IDENTIFIER)
        
        return Token(token_type, id_str, start_line, start_col)
    
    def read_string(self) -> Token:
        """Read string literal (for print statements)"""
        start_line = self.line
        start_col = self.column
        
        # Skip opening quote
        self.advance()
        
        string_value = ''
        while self.current_char and self.current_char != '"':
            if self.current_char == '\n':
                self.error("Unterminated string literal")
            string_value += self.current_char
            self.advance()
        
        if self.current_char != '"':
            self.error("Unterminated string literal")
        
        # Skip closing quote
        self.advance()
        
        return Token(TokenType.STRING, string_value, start_line, start_col)
    
    def get_next_token(self) -> Token:
        """Get the next token from source code"""
        
        while self.current_char:
            
            # Skip whitespace
            if self.current_char in ' \t\r\n':
                self.skip_whitespace()
                continue
            
            # Skip comments
            if self.current_char == '/' and self.peek() == '/':
                self.skip_comment()
                continue
            
            # Integer literal
            if self.current_char.isdigit():
                return self.read_integer()
            
            # Identifier or keyword
            if self.current_char.isalpha() and self.current_char.islower():
                return self.read_identifier()
            
            # String literal
            if self.current_char == '"':
                return self.read_string()
            
            # Two-character operators
            if self.current_char == '=' and self.peek() == '=':
                token = Token(TokenType.EQUAL, '==', self.line, self.column)
                self.advance()
                self.advance()
                return token
            
            if self.current_char == '!' and self.peek() == '=':
                token = Token(TokenType.NOT_EQUAL, '!=', self.line, self.column)
                self.advance()
                self.advance()
                return token
            
            # Single-character tokens
            single_char_tokens = {
                '=': TokenType.ASSIGN,
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MULTIPLY,
                '/': TokenType.DIVIDE,
                '>': TokenType.GREATER,
                '<': TokenType.LESS,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                ';': TokenType.SEMICOLON,
                '.': TokenType.DOT,
                ',': TokenType.COMMA,
            }
            
            if self.current_char in single_char_tokens:
                token = Token(
                    single_char_tokens[self.current_char],
                    self.current_char,
                    self.line,
                    self.column
                )
                self.advance()
                return token
            
            # Unknown character
            self.error(f"Unknown character '{self.current_char}'")
        
        # End of file
        return Token(TokenType.EOF, None, self.line, self.column)
    
    def tokenize(self) -> List[Token]:
        """Tokenize entire source code"""
        tokens = []
        
        while True:
            token = self.get_next_token()
            tokens.append(token)
            
            if token.type == TokenType.EOF:
                break
        
        self.tokens = tokens
        return tokens


# ============================================================================
# TESTING THE LEXER
# ============================================================================

if __name__ == "__main__":
    # Test code
    test_code = """
    player hero {
        x = 0;
        y = 0;
        health = 100;
    }
    
    enemy monster {
        x = 5;
        y = 5;
    }
    
    move hero right 5;
    move hero up 5;
    
    if hero.x == monster.x {
        set hero.health = hero.health - 10;
        print "Hit by monster!";
    }
    """
    
    print("=" * 60)
    print("TINYGAME LEXER TEST")
    print("=" * 60)
    
    lexer = Lexer(test_code)
    tokens = lexer.tokenize()
    
    print("\nTokens generated:")
    print("-" * 60)
    for token in tokens:
        print(token)
    
    print("\n" + "=" * 60)
    print(f"Total tokens: {len(tokens)}")
    print("=" * 60)