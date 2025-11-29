"""
TinyGame Compiler - Semantic Analyzer
Performs type checking, scope management, and builds symbol table
"""

from parser import *
from lexer import Lexer
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

# ============================================================================
# SYMBOL TABLE STRUCTURES
# ============================================================================

@dataclass
class Symbol:
    """Represents a symbol in the symbol table"""
    name: str
    symbol_type: str  # 'entity', 'property'
    data_type: str    # 'int', 'player', 'enemy'
    value: Optional[int] = None
    line: int = 0
    scope: str = "global"


@dataclass
class EntitySymbol:
    """Represents an entity (player or enemy)"""
    name: str
    entity_type: str  # 'player' or 'enemy'
    properties: Dict[str, Symbol] = field(default_factory=dict)
    line: int = 0


class SymbolTable:
    """Symbol table for managing entities and their properties"""
    
    def __init__(self):
        """Initialize empty symbol table"""
        self.entities: Dict[str, EntitySymbol] = {}
        self.current_scope: Optional[str] = None
    
    def add_entity(self, name: str, entity_type: str, line: int) -> bool:
        """Add entity to symbol table"""
        if name in self.entities:
            return False
        
        self.entities[name] = EntitySymbol(name, entity_type, {}, line)
        return True
    
    def entity_exists(self, name: str) -> bool:
        """Check if entity exists"""
        return name in self.entities
    
    def add_property(self, entity_name: str, prop_name: str, 
                    value: Optional[int], line: int) -> bool:
        """Add property to an entity"""
        if entity_name not in self.entities:
            return False
        
        entity = self.entities[entity_name]
        
        if prop_name in entity.properties:
            return False
        
        symbol = Symbol(
            name=prop_name,
            symbol_type='property',
            data_type='int',
            value=value,
            line=line,
            scope=entity_name
        )
        
        entity.properties[prop_name] = symbol
        return True
    
    def property_exists(self, entity_name: str, prop_name: str) -> bool:
        """Check if property exists for an entity"""
        if entity_name not in self.entities:
            return False
        return prop_name in self.entities[entity_name].properties
    
    def get_entity(self, name: str) -> Optional[EntitySymbol]:
        """Get entity symbol"""
        return self.entities.get(name)
    
    def get_property(self, entity_name: str, prop_name: str) -> Optional[Symbol]:
        """Get property symbol"""
        if entity_name in self.entities:
            return self.entities[entity_name].properties.get(prop_name)
        return None
    
    def __str__(self) -> str:
        """String representation of symbol table"""
        result = "=" * 70 + "\n"
        result += "SYMBOL TABLE\n"
        result += "=" * 70 + "\n\n"
        
        result += "GLOBAL SCOPE - ENTITIES:\n"
        result += "-" * 70 + "\n"
        result += f"{'Name':<15} {'Type':<10} {'Line':<10} {'Properties'}\n"
        result += "-" * 70 + "\n"
        
        for entity_name, entity in self.entities.items():
            prop_count = len(entity.properties)
            result += f"{entity_name:<15} {entity.entity_type:<10} {entity.line:<10} {prop_count} properties\n"
        
        result += "\n"
        
        # Print properties for each entity
        for entity_name, entity in self.entities.items():
            result += f"\nENTITY SCOPE: {entity_name}\n"
            result += "-" * 70 + "\n"
            result += f"{'Property':<15} {'Type':<10} {'Value':<10} {'Line'}\n"
            result += "-" * 70 + "\n"
            
            for prop_name, prop_symbol in entity.properties.items():
                value_str = str(prop_symbol.value) if prop_symbol.value is not None else "None"
                result += f"{prop_name:<15} {prop_symbol.data_type:<10} {value_str:<10} {prop_symbol.line}\n"
        
        result += "\n" + "=" * 70 + "\n"
        return result


# ============================================================================
# SEMANTIC ANALYZER
# ============================================================================

class SemanticAnalyzer:
    """Performs semantic analysis on AST"""
    
    def __init__(self, ast: Program):
        """Initialize semantic analyzer"""
        self.ast = ast
        self.symbol_table = SymbolTable()
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def error(self, msg: str, line: int, col: int):
        """Record semantic error"""
        self.errors.append(f"Semantic Error at {line}:{col}: {msg}")
    
    def warning(self, msg: str, line: int, col: int):
        """Record semantic warning"""
        self.warnings.append(f"Warning at {line}:{col}: {msg}")
    
    def analyze(self) -> bool:
        """Perform complete semantic analysis"""
        # Phase 1: Build symbol table from entity declarations
        self.build_symbol_table()
        
        # Phase 2: Check statements for semantic correctness
        if not self.errors:
            self.check_statements()
        
        # Return True if no errors
        return len(self.errors) == 0
    
    def build_symbol_table(self):
        """Build symbol table from entity declarations"""
        for entity in self.ast.entities:
            # Check for duplicate entity names
            if not self.symbol_table.add_entity(
                entity.name, 
                entity.entity_type, 
                entity.line
            ):
                self.error(
                    f"Entity '{entity.name}' is already declared",
                    entity.line,
                    entity.column
                )
                continue
            
            # Add properties to entity
            property_names: Set[str] = set()
            
            for assignment in entity.properties:
                prop_name = assignment.var_name
                
                # Check for duplicate properties
                if prop_name in property_names:
                    self.error(
                        f"Property '{prop_name}' is already declared in entity '{entity.name}'",
                        assignment.line,
                        assignment.column
                    )
                    continue
                
                property_names.add(prop_name)
                
                # Evaluate initial value (if it's a constant)
                value = self.evaluate_constant_expression(assignment.expression)
                
                # Check expression type
                expr_type = self.check_expression(assignment.expression, entity.name)
                if expr_type != 'int':
                    self.error(
                        f"Property '{prop_name}' must be of type int, got {expr_type}",
                        assignment.line,
                        assignment.column
                    )
                
                # Add property to symbol table
                self.symbol_table.add_property(
                    entity.name,
                    prop_name,
                    value,
                    assignment.line
                )
    
    def check_statements(self):
        """Check all statements for semantic correctness"""
        for stmt in self.ast.statements:
            self.check_statement(stmt)
    
    def check_statement(self, stmt: Statement):
        """Check individual statement"""
        if isinstance(stmt, MoveStatement):
            self.check_move_statement(stmt)
        elif isinstance(stmt, SetStatement):
            self.check_set_statement(stmt)
        elif isinstance(stmt, IfStatement):
            self.check_if_statement(stmt)
        elif isinstance(stmt, PrintStatement):
            # Print statements are always valid
            pass
    
    def check_move_statement(self, stmt: MoveStatement):
        """Check move statement semantics"""
        # Check if entity exists
        if not self.symbol_table.entity_exists(stmt.entity_name):
            self.error(
                f"Entity '{stmt.entity_name}' is not declared",
                stmt.line,
                stmt.column
            )
            return
        
        # Check if amount expression is valid and evaluates to int
        expr_type = self.check_expression(stmt.amount, None)
        if expr_type != 'int':
            self.error(
                f"Move amount must be an integer, got {expr_type}",
                stmt.line,
                stmt.column
            )
    
    def check_set_statement(self, stmt: SetStatement):
        """Check set statement semantics"""
        # Check if entity exists
        if not self.symbol_table.entity_exists(stmt.entity_name):
            self.error(
                f"Entity '{stmt.entity_name}' is not declared",
                stmt.line,
                stmt.column
            )
            return
        
        # Check if property exists
        if not self.symbol_table.property_exists(stmt.entity_name, stmt.property_name):
            self.error(
                f"Entity '{stmt.entity_name}' has no property '{stmt.property_name}'",
                stmt.line,
                stmt.column
            )
            return
        
        # Check if expression is valid and evaluates to int
        expr_type = self.check_expression(stmt.expression, None)
        if expr_type != 'int':
            self.error(
                f"Assignment value must be an integer, got {expr_type}",
                stmt.line,
                stmt.column
            )
    
    def check_if_statement(self, stmt: IfStatement):
        """Check if statement semantics"""
        # Check condition
        self.check_condition(stmt.condition)
        
        # Check body statements
        for body_stmt in stmt.body:
            self.check_statement(body_stmt)
    
    def check_condition(self, cond: Condition):
        """Check condition semantics"""
        # Check both sides of condition
        left_type = self.check_expression(cond.left, None)
        right_type = self.check_expression(cond.right, None)
        
        # Both sides must be int
        if left_type != 'int' or right_type != 'int':
            self.error(
                f"Condition operands must be integers, got {left_type} and {right_type}",
                cond.line,
                cond.column
            )
    
    def check_expression(self, expr: Expression, current_entity: Optional[str]) -> str:
        """
        Check expression and return its type
        Returns: 'int' or 'error'
        """
        if isinstance(expr, IntegerLiteral):
            return 'int'
        
        elif isinstance(expr, Identifier):
            # Identifiers alone are not allowed in TinyGame
            # Only property access is allowed
            self.error(
                f"Invalid identifier '{expr.name}'. Use entity.property syntax",
                expr.line,
                expr.column
            )
            return 'error'
        
        elif isinstance(expr, PropertyAccess):
            # Check if entity exists
            if not self.symbol_table.entity_exists(expr.entity_name):
                self.error(
                    f"Entity '{expr.entity_name}' is not declared",
                    expr.line,
                    expr.column
                )
                return 'error'
            
            # Check if property exists
            if not self.symbol_table.property_exists(expr.entity_name, expr.property_name):
                self.error(
                    f"Entity '{expr.entity_name}' has no property '{expr.property_name}'",
                    expr.line,
                    expr.column
                )
                return 'error'
            
            return 'int'
        
        elif isinstance(expr, BinaryOp):
            left_type = self.check_expression(expr.left, current_entity)
            right_type = self.check_expression(expr.right, current_entity)
            
            if left_type == 'int' and right_type == 'int':
                return 'int'
            else:
                self.error(
                    f"Binary operation requires integer operands",
                    expr.line,
                    expr.column
                )
                return 'error'
        
        return 'error'
    
    def evaluate_constant_expression(self, expr: Expression) -> Optional[int]:
        """
        Try to evaluate expression at compile time
        Returns value if constant, None otherwise
        """
        if isinstance(expr, IntegerLiteral):
            return expr.value
        
        elif isinstance(expr, BinaryOp):
            left_val = self.evaluate_constant_expression(expr.left)
            right_val = self.evaluate_constant_expression(expr.right)
            
            if left_val is not None and right_val is not None:
                if expr.operator == '+':
                    return left_val + right_val
                elif expr.operator == '-':
                    return left_val - right_val
                elif expr.operator == '*':
                    return left_val * right_val
                elif expr.operator == '/':
                    if right_val != 0:
                        return left_val // right_val
        
        return None
    
    def print_results(self):
        """Print analysis results"""
        print("\n" + "=" * 70)
        print("SEMANTIC ANALYSIS RESULTS")
        print("=" * 70)
        
        if self.errors:
            print("\nERRORS FOUND:")
            print("-" * 70)
            for error in self.errors:
                print(f"  ❌ {error}")
        else:
            print("\n✅ No semantic errors found!")
        
        if self.warnings:
            print("\nWARNINGS:")
            print("-" * 70)
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
        
        print("\n" + "=" * 70)


# ============================================================================
# TESTING THE SEMANTIC ANALYZER
# ============================================================================

if __name__ == "__main__":
    # Test code with semantic errors
    test_code_1 = """
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
    
    # Test code with errors
    test_code_2 = """
    player hero {
        x = 0;
        y = 0;
    }
    
    // This should cause errors
    move ghost right 5;
    set hero.z = 10;
    """
    
    print("=" * 70)
    print("TINYGAME SEMANTIC ANALYZER TEST")
    print("=" * 70)
    
    # Test 1: Valid code
    print("\n" + "=" * 70)
    print("TEST 1: Valid Code")
    print("=" * 70)
    
    lexer1 = Lexer(test_code_1)
    tokens1 = lexer1.tokenize()
    parser1 = Parser(tokens1)
    ast1 = parser1.parse()
    
    analyzer1 = SemanticAnalyzer(ast1)
    success1 = analyzer1.analyze()
    
    print(analyzer1.symbol_table)
    analyzer1.print_results()
    
    # Test 2: Code with errors
    print("\n" + "=" * 70)
    print("TEST 2: Code with Semantic Errors")
    print("=" * 70)
    
    lexer2 = Lexer(test_code_2)
    tokens2 = lexer2.tokenize()
    parser2 = Parser(tokens2)
    ast2 = parser2.parse()
    
    analyzer2 = SemanticAnalyzer(ast2)
    success2 = analyzer2.analyze()
    
    print(analyzer2.symbol_table)
    analyzer2.print_results()
    
    print("\n" + "=" * 70)