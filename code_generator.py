"""
TinyGame Compiler - Intermediate Code Generator
Generates three-address code (TAC) from AST
"""

from parser import *
from semantic_analyzer import SemanticAnalyzer, SymbolTable
from typing import List, Optional
from dataclasses import dataclass

# ============================================================================
# THREE-ADDRESS CODE INSTRUCTION
# ============================================================================

@dataclass
class TACInstruction:
    """Represents a single three-address code instruction"""
    op: str              # Operation: =, +, -, *, /, ==, !=, >, <, goto, if, label, print, etc.
    arg1: Optional[str]  # First argument
    arg2: Optional[str]  # Second argument
    result: Optional[str] # Result
    
    def __str__(self) -> str:
        """String representation of instruction"""
        if self.op == 'label':
            return f"{self.result}:"
        elif self.op == 'goto':
            return f"    goto {self.result}"
        elif self.op == 'if':
            return f"    if {self.arg1} goto {self.result}"
        elif self.op == 'ifFalse':
            return f"    ifFalse {self.arg1} goto {self.result}"
        elif self.op == 'print':
            return f"    print {self.arg1}"
        elif self.op == 'move':
            # move entity direction amount
            return f"    move {self.arg1} {self.arg2} {self.result}"
        elif self.op == '=':
            if self.arg1 is None:
                return f"    {self.result} = (uninitialized)"
            return f"    {self.result} = {self.arg1}"
        elif self.op in ['+', '-', '*', '/']:
            return f"    {self.result} = {self.arg1} {self.op} {self.arg2}"
        elif self.op in ['==', '!=', '>', '<']:
            return f"    {self.result} = {self.arg1} {self.op} {self.arg2}"
        else:
            return f"    {self.op} {self.arg1} {self.arg2} {self.result}"


# ============================================================================
# INTERMEDIATE CODE GENERATOR
# ============================================================================

class IntermediateCodeGenerator:
    """Generates three-address code from AST"""
    
    def __init__(self, ast: Program, symbol_table: SymbolTable):
        """Initialize code generator"""
        self.ast = ast
        self.symbol_table = symbol_table
        self.instructions: List[TACInstruction] = []
        self.temp_counter = 0
        self.label_counter = 0
    
    def new_temp(self) -> str:
        """Generate new temporary variable"""
        temp = f"t{self.temp_counter}"
        self.temp_counter += 1
        return temp
    
    def new_label(self) -> str:
        """Generate new label"""
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label
    
    def emit(self, op: str, arg1: Optional[str] = None, 
             arg2: Optional[str] = None, result: Optional[str] = None):
        """Emit a three-address code instruction"""
        instruction = TACInstruction(op, arg1, arg2, result)
        self.instructions.append(instruction)
    
    def generate(self) -> List[TACInstruction]:
        """Generate intermediate code for entire program"""
        # Generate code for entity initializations
        for entity in self.ast.entities:
            self.generate_entity(entity)
        
        # Generate code for statements
        for stmt in self.ast.statements:
            self.generate_statement(stmt)
        
        return self.instructions
    
    def generate_entity(self, entity: Entity):
        """Generate code for entity initialization"""
        self.emit('label', result=f"init_{entity.name}")
        
        for assignment in entity.properties:
            # Generate code for initial value
            value_temp = self.generate_expression(assignment.expression)
            
            # Assign to property
            property_name = f"{entity.name}.{assignment.var_name}"
            self.emit('=', value_temp, None, property_name)
    
    def generate_statement(self, stmt: Statement):
        """Generate code for a statement"""
        if isinstance(stmt, MoveStatement):
            self.generate_move_statement(stmt)
        elif isinstance(stmt, SetStatement):
            self.generate_set_statement(stmt)
        elif isinstance(stmt, IfStatement):
            self.generate_if_statement(stmt)
        elif isinstance(stmt, PrintStatement):
            self.generate_print_statement(stmt)
    
    def generate_move_statement(self, stmt: MoveStatement):
        """Generate code for move statement"""
        # Evaluate amount
        amount_temp = self.generate_expression(stmt.amount)
        
        # Emit move instruction
        # Format: move entity direction amount
        self.emit('move', stmt.entity_name, stmt.direction, amount_temp)
        
        # Update entity position properties based on direction
        # This is done by the interpreter, but we can add TAC for clarity
        
        if stmt.direction in ['left', 'right']:
            # Get current x
            current_x = f"{stmt.entity_name}.x"
            temp_x = self.new_temp()
            
            if stmt.direction == 'right':
                self.emit('+', current_x, amount_temp, temp_x)
            else:  # left
                self.emit('-', current_x, amount_temp, temp_x)
            
            self.emit('=', temp_x, None, current_x)
        
        elif stmt.direction in ['up', 'down']:
            # Get current y
            current_y = f"{stmt.entity_name}.y"
            temp_y = self.new_temp()
            
            if stmt.direction == 'up':
                self.emit('+', current_y, amount_temp, temp_y)
            else:  # down
                self.emit('-', current_y, amount_temp, temp_y)
            
            self.emit('=', temp_y, None, current_y)
    
    def generate_set_statement(self, stmt: SetStatement):
        """Generate code for set statement"""
        # Evaluate expression
        value_temp = self.generate_expression(stmt.expression)
        
        # Assign to property
        property_name = f"{stmt.entity_name}.{stmt.property_name}"
        self.emit('=', value_temp, None, property_name)
    
    def generate_if_statement(self, stmt: IfStatement):
        """Generate code for if statement"""
        # Generate condition evaluation
        condition_temp = self.generate_condition(stmt.condition)
        
        # Create labels
        true_label = self.new_label()
        end_label = self.new_label()
        
        # If condition is true, goto true_label
        self.emit('if', condition_temp, None, true_label)
        
        # If false, goto end_label
        self.emit('goto', None, None, end_label)
        
        # True branch
        self.emit('label', result=true_label)
        for body_stmt in stmt.body:
            self.generate_statement(body_stmt)
        
        # End label
        self.emit('label', result=end_label)
    
    def generate_print_statement(self, stmt: PrintStatement):
        """Generate code for print statement"""
        self.emit('print', f'"{stmt.message}"', None, None)
    
    def generate_condition(self, cond: Condition) -> str:
        """Generate code for condition and return result temp"""
        # Evaluate left and right expressions
        left_temp = self.generate_expression(cond.left)
        right_temp = self.generate_expression(cond.right)
        
        # Create temporary for result
        result_temp = self.new_temp()
        
        # Emit comparison
        self.emit(cond.operator, left_temp, right_temp, result_temp)
        
        return result_temp
    
    def generate_expression(self, expr: Expression) -> str:
        """Generate code for expression and return result temp/value"""
        if isinstance(expr, IntegerLiteral):
            # Return literal value directly
            return str(expr.value)
        
        elif isinstance(expr, Identifier):
            # Should not happen in valid TinyGame code
            return expr.name
        
        elif isinstance(expr, PropertyAccess):
            # Return property reference
            return f"{expr.entity_name}.{expr.property_name}"
        
        elif isinstance(expr, BinaryOp):
            # Evaluate left and right
            left_temp = self.generate_expression(expr.left)
            right_temp = self.generate_expression(expr.right)
            
            # Create temporary for result
            result_temp = self.new_temp()
            
            # Emit operation
            self.emit(expr.operator, left_temp, right_temp, result_temp)
            
            return result_temp
        
        return "error"
    
    def print_code(self):
        """Print generated three-address code"""
        print("\n" + "=" * 70)
        print("THREE-ADDRESS CODE (INTERMEDIATE REPRESENTATION)")
        print("=" * 70)
        print()
        
        for i, instruction in enumerate(self.instructions):
            print(f"{i:3d}  {instruction}")
        
        print("\n" + "=" * 70)
        print(f"Total instructions: {len(self.instructions)}")
        print("=" * 70)


# ============================================================================
# TESTING THE CODE GENERATOR
# ============================================================================

if __name__ == "__main__":
    from lexer import Lexer
    from parser import Parser
    
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
    
    print("=" * 70)
    print("TINYGAME INTERMEDIATE CODE GENERATOR TEST")
    print("=" * 70)
    
    # Lexical analysis
    print("\n[1] Lexical Analysis...")
    lexer = Lexer(test_code)
    tokens = lexer.tokenize()
    print(f"✓ Generated {len(tokens)} tokens")
    
    # Syntax analysis
    print("\n[2] Syntax Analysis...")
    parser = Parser(tokens)
    ast = parser.parse()
    print("✓ AST built successfully")
    
    # Semantic analysis
    print("\n[3] Semantic Analysis...")
    analyzer = SemanticAnalyzer(ast)
    if analyzer.analyze():
        print("✓ No semantic errors")
    else:
        print("✗ Semantic errors found:")
        for error in analyzer.errors:
            print(f"  {error}")
        exit(1)
    
    # Intermediate code generation
    print("\n[4] Intermediate Code Generation...")
    code_gen = IntermediateCodeGenerator(ast, analyzer.symbol_table)
    tac_instructions = code_gen.generate()
    print(f"✓ Generated {len(tac_instructions)} TAC instructions")
    
    # Print the code
    code_gen.print_code()
    
    # Print symbol table for reference
    print("\n")
    print(analyzer.symbol_table)