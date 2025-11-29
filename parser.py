"""
TinyGame Compiler - Syntax Analyzer (Parser)
Builds Abstract Syntax Tree (AST) from tokens
"""

from lexer import Token, TokenType, Lexer
from typing import List, Optional, Any
from dataclasses import dataclass

# ============================================================================
# AST NODE CLASSES
# ============================================================================

@dataclass
class ASTNode:
    """Base class for all AST nodes"""
    line: int
    column: int


@dataclass
class Program(ASTNode):
    """Root node of the AST"""
    entities: List['Entity']
    statements: List['Statement']


@dataclass
class Entity(ASTNode):
    """Entity declaration (player or enemy)"""
    entity_type: str  # 'player' or 'enemy'
    name: str
    properties: List['Assignment']


@dataclass
class Assignment(ASTNode):
    """Property assignment: x = 5"""
    var_name: str
    expression: 'Expression'


@dataclass
class Statement(ASTNode):
    """Base class for statements"""
    pass


@dataclass
class MoveStatement(Statement):
    """Move statement: move hero right 5"""
    entity_name: str
    direction: str  # 'up', 'down', 'left', 'right'
    amount: 'Expression'


@dataclass
class SetStatement(Statement):
    """Set statement: set hero.x = 5"""
    entity_name: str
    property_name: str
    expression: 'Expression'


@dataclass
class IfStatement(Statement):
    """If statement with condition and body"""
    condition: 'Condition'
    body: List[Statement]


@dataclass
class PrintStatement(Statement):
    """Print statement: print "message" """
    message: str


@dataclass
class Condition(ASTNode):
    """Conditional expression: x == y"""
    left: 'Expression'
    operator: str  # '==', '!=', '>', '<'
    right: 'Expression'


@dataclass
class Expression(ASTNode):
    """Base class for expressions"""
    pass


@dataclass
class BinaryOp(Expression):
    """Binary operation: left op right"""
    left: Expression
    operator: str  # '+', '-', '*', '/'
    right: Expression


@dataclass
class IntegerLiteral(Expression):
    """Integer literal: 42"""
    value: int


@dataclass
class Identifier(Expression):
    """Simple identifier: hero"""
    name: str


@dataclass
class PropertyAccess(Expression):
    """Property access: hero.x"""
    entity_name: str
    property_name: str


# ============================================================================
# PARSER CLASS
# ============================================================================

class Parser:
    """Recursive Descent Parser for TinyGame"""
    
    def __init__(self, tokens: List[Token]):
        """Initialize parser with token list"""
        self.tokens = tokens
        self.pos = 0
        self.current_token = tokens[0] if tokens else None
    
    def error(self, msg: str):
        """Raise syntax error"""
        if self.current_token:
            raise Exception(
                f"Syntax Error at {self.current_token.line}:{self.current_token.column}: {msg}"
            )
        else:
            raise Exception(f"Syntax Error: {msg}")
    
    def advance(self):
        """Move to next token"""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None
    
    def peek(self, offset: int = 1) -> Optional[Token]:
        """Look ahead at next token(s)"""
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None
    
    def expect(self, token_type: TokenType) -> Token:
        """Consume token of expected type or raise error"""
        if not self.current_token or self.current_token.type != token_type:
            self.error(f"Expected {token_type.name}, got {self.current_token.type.name if self.current_token else 'EOF'}")
        token = self.current_token
        self.advance()
        return token
    
    def match(self, *token_types: TokenType) -> bool:
        """Check if current token matches any of given types"""
        if not self.current_token:
            return False
        return self.current_token.type in token_types
    
    # ========================================================================
    # GRAMMAR RULES
    # ========================================================================
    
    def parse(self) -> Program:
        """
        <program> ::= <entity>* <statement>*
        """
        line = self.current_token.line if self.current_token else 1
        col = self.current_token.column if self.current_token else 1
        
        entities = []
        statements = []
        
        # Parse entities
        while self.match(TokenType.PLAYER, TokenType.ENEMY):
            entities.append(self.parse_entity())
        
        # Parse statements
        while not self.match(TokenType.EOF):
            statements.append(self.parse_statement())
        
        return Program(line, col, entities, statements)
    
    def parse_entity(self) -> Entity:
        """
        <entity> ::= ("player" | "enemy") ID "{" <assign>* "}"
        """
        line = self.current_token.line
        col = self.current_token.column
        
        # Get entity type
        if self.match(TokenType.PLAYER):
            entity_type = 'player'
            self.advance()
        elif self.match(TokenType.ENEMY):
            entity_type = 'enemy'
            self.advance()
        else:
            self.error("Expected 'player' or 'enemy'")
        
        # Get entity name
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value
        
        # Parse property block
        self.expect(TokenType.LBRACE)
        
        properties = []
        while not self.match(TokenType.RBRACE):
            properties.append(self.parse_assignment())
        
        self.expect(TokenType.RBRACE)
        
        return Entity(line, col, entity_type, name, properties)
    
    def parse_assignment(self) -> Assignment:
        """
        <assign> ::= ID "=" <expr> ";"
        """
        line = self.current_token.line
        col = self.current_token.column
        
        var_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        expr = self.parse_expression()
        self.expect(TokenType.SEMICOLON)
        
        return Assignment(line, col, var_name, expr)
    
    def parse_statement(self) -> Statement:
        """
        <statement> ::= <move_stmt> | <set_stmt> | <if_stmt> | <print_stmt>
        """
        if self.match(TokenType.MOVE):
            return self.parse_move_statement()
        elif self.match(TokenType.SET):
            return self.parse_set_statement()
        elif self.match(TokenType.IF):
            return self.parse_if_statement()
        elif self.match(TokenType.PRINT):
            return self.parse_print_statement()
        else:
            self.error(f"Unexpected token: {self.current_token.type.name}")
    
    def parse_move_statement(self) -> MoveStatement:
        """
        <move_stmt> ::= "move" ID <direction> <expr> ";"
        <direction> ::= "up" | "down" | "left" | "right"
        """
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.MOVE)
        entity_name = self.expect(TokenType.IDENTIFIER).value
        
        # Get direction
        if self.match(TokenType.UP):
            direction = 'up'
        elif self.match(TokenType.DOWN):
            direction = 'down'
        elif self.match(TokenType.LEFT):
            direction = 'left'
        elif self.match(TokenType.RIGHT):
            direction = 'right'
        else:
            self.error("Expected direction (up, down, left, right)")
        
        self.advance()
        
        amount = self.parse_expression()
        self.expect(TokenType.SEMICOLON)
        
        return MoveStatement(line, col, entity_name, direction, amount)
    
    def parse_set_statement(self) -> SetStatement:
        """
        <set_stmt> ::= "set" ID "." ID "=" <expr> ";"
        """
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.SET)
        entity_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.DOT)
        property_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        expr = self.parse_expression()
        self.expect(TokenType.SEMICOLON)
        
        return SetStatement(line, col, entity_name, property_name, expr)
    
    def parse_if_statement(self) -> IfStatement:
        """
        <if_stmt> ::= "if" <condition> "{" <statement>* "}"
        """
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.IF)
        condition = self.parse_condition()
        self.expect(TokenType.LBRACE)
        
        body = []
        while not self.match(TokenType.RBRACE):
            body.append(self.parse_statement())
        
        self.expect(TokenType.RBRACE)
        
        return IfStatement(line, col, condition, body)
    
    def parse_print_statement(self) -> PrintStatement:
        """
        <print_stmt> ::= "print" STRING ";"
        """
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.PRINT)
        message = self.expect(TokenType.STRING).value
        self.expect(TokenType.SEMICOLON)
        
        return PrintStatement(line, col, message)
    
    def parse_condition(self) -> Condition:
        """
        <condition> ::= <expr> <relop> <expr>
        <relop> ::= "==" | "!=" | ">" | "<"
        """
        line = self.current_token.line
        col = self.current_token.column
        
        left = self.parse_expression()
        
        if self.match(TokenType.EQUAL):
            operator = '=='
        elif self.match(TokenType.NOT_EQUAL):
            operator = '!='
        elif self.match(TokenType.GREATER):
            operator = '>'
        elif self.match(TokenType.LESS):
            operator = '<'
        else:
            self.error("Expected relational operator (==, !=, >, <)")
        
        self.advance()
        right = self.parse_expression()
        
        return Condition(line, col, left, operator, right)
    
    def parse_expression(self) -> Expression:
        """
        <expr> ::= <term> ( ("+" | "-") <term> )*
        """
        left = self.parse_term()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = '+' if self.current_token.type == TokenType.PLUS else '-'
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            right = self.parse_term()
            left = BinaryOp(line, col, left, operator, right)
        
        return left
    
    def parse_term(self) -> Expression:
        """
        <term> ::= <factor> ( ("*" | "/") <factor> )*
        """
        left = self.parse_factor()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE):
            operator = '*' if self.current_token.type == TokenType.MULTIPLY else '/'
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            right = self.parse_factor()
            left = BinaryOp(line, col, left, operator, right)
        
        return left
    
    def parse_factor(self) -> Expression:
        """
        <factor> ::= INTEGER | ID | ID "." ID | "(" <expr> ")"
        """
        line = self.current_token.line
        col = self.current_token.column
        
        # Integer literal
        if self.match(TokenType.INTEGER):
            value = self.current_token.value
            self.advance()
            return IntegerLiteral(line, col, value)
        
        # Identifier or property access
        if self.match(TokenType.IDENTIFIER):
            name = self.current_token.value
            self.advance()
            
            # Check for property access
            if self.match(TokenType.DOT):
                self.advance()
                property_name = self.expect(TokenType.IDENTIFIER).value
                return PropertyAccess(line, col, name, property_name)
            
            return Identifier(line, col, name)
        
        # Parenthesized expression
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        self.error(f"Unexpected token in expression: {self.current_token.type.name}")


# ============================================================================
# PRETTY PRINTER FOR AST
# ============================================================================

class ASTPrinter:
    """Utility to print AST in readable format"""
    
    @staticmethod
    def print_ast(node: ASTNode, indent: int = 0) -> str:
        """Recursively print AST node"""
        prefix = "  " * indent
        
        if isinstance(node, Program):
            result = f"{prefix}Program\n"
            result += f"{prefix}  Entities:\n"
            for entity in node.entities:
                result += ASTPrinter.print_ast(entity, indent + 2)
            result += f"{prefix}  Statements:\n"
            for stmt in node.statements:
                result += ASTPrinter.print_ast(stmt, indent + 2)
            return result
        
        elif isinstance(node, Entity):
            result = f"{prefix}{node.entity_type.capitalize()}: {node.name}\n"
            for prop in node.properties:
                result += ASTPrinter.print_ast(prop, indent + 1)
            return result
        
        elif isinstance(node, Assignment):
            result = f"{prefix}{node.var_name} = "
            result += ASTPrinter.print_ast(node.expression, 0).strip() + "\n"
            return result
        
        elif isinstance(node, MoveStatement):
            result = f"{prefix}Move {node.entity_name} {node.direction} "
            result += ASTPrinter.print_ast(node.amount, 0).strip() + "\n"
            return result
        
        elif isinstance(node, SetStatement):
            result = f"{prefix}Set {node.entity_name}.{node.property_name} = "
            result += ASTPrinter.print_ast(node.expression, 0).strip() + "\n"
            return result
        
        elif isinstance(node, IfStatement):
            result = f"{prefix}If "
            result += ASTPrinter.print_ast(node.condition, 0).strip() + "\n"
            for stmt in node.body:
                result += ASTPrinter.print_ast(stmt, indent + 1)
            return result
        
        elif isinstance(node, PrintStatement):
            return f"{prefix}Print \"{node.message}\"\n"
        
        elif isinstance(node, Condition):
            left = ASTPrinter.print_ast(node.left, 0).strip()
            right = ASTPrinter.print_ast(node.right, 0).strip()
            return f"({left} {node.operator} {right})"
        
        elif isinstance(node, BinaryOp):
            left = ASTPrinter.print_ast(node.left, 0).strip()
            right = ASTPrinter.print_ast(node.right, 0).strip()
            return f"({left} {node.operator} {right})"
        
        elif isinstance(node, IntegerLiteral):
            return f"{node.value}"
        
        elif isinstance(node, Identifier):
            return f"{node.name}"
        
        elif isinstance(node, PropertyAccess):
            return f"{node.entity_name}.{node.property_name}"
        
        return f"{prefix}Unknown node: {type(node).__name__}\n"


# ============================================================================
# TESTING THE PARSER
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
    print("TINYGAME PARSER TEST")
    print("=" * 60)
    
    # Lexical analysis
    print("\n[1] Tokenizing...")
    lexer = Lexer(test_code)
    tokens = lexer.tokenize()
    print(f"Generated {len(tokens)} tokens")
    
    # Syntax analysis
    print("\n[2] Parsing...")
    parser = Parser(tokens)
    ast = parser.parse()
    print("AST built successfully!")
    
    # Print AST
    print("\n[3] Abstract Syntax Tree:")
    print("-" * 60)
    print(ASTPrinter.print_ast(ast))
    
    print("=" * 60)