"""
TinyGame Compiler - Interpreter
Executes optimized three-address code
"""

from code_generator import TACInstruction
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# ============================================================================
# RUNTIME ENVIRONMENT
# ============================================================================

@dataclass
class Entity:
    """Runtime representation of an entity"""
    name: str
    entity_type: str  # 'player' or 'enemy'
    properties: Dict[str, int] = field(default_factory=dict)
    
    def __str__(self) -> str:
        props = ', '.join([f"{k}={v}" for k, v in self.properties.items()])
        return f"{self.entity_type} {self.name}: {props}"


class RuntimeEnvironment:
    """Manages runtime state during execution"""
    
    def __init__(self):
        """Initialize runtime environment"""
        self.entities: Dict[str, Entity] = {}
        self.temporaries: Dict[str, int] = {}
        self.output: List[str] = []
    
    def create_entity(self, name: str, entity_type: str):
        """Create a new entity"""
        self.entities[name] = Entity(name, entity_type)
    
    def set_property(self, entity_name: str, prop_name: str, value: int):
        """Set entity property value"""
        if entity_name not in self.entities:
            self.entities[entity_name] = Entity(entity_name, 'unknown')
        self.entities[entity_name].properties[prop_name] = value
    
    def get_property(self, entity_name: str, prop_name: str) -> Optional[int]:
        """Get entity property value"""
        if entity_name in self.entities:
            return self.entities[entity_name].properties.get(prop_name)
        return None
    
    def set_temp(self, temp_name: str, value: int):
        """Set temporary variable value"""
        self.temporaries[temp_name] = value
    
    def get_temp(self, temp_name: str) -> Optional[int]:
        """Get temporary variable value"""
        return self.temporaries.get(temp_name)
    
    def add_output(self, message: str):
        """Add to output"""
        self.output.append(message)
    
    def print_state(self):
        """Print current runtime state"""
        print("\n" + "=" * 70)
        print("RUNTIME STATE")
        print("=" * 70)
        
        print("\nEntities:")
        print("-" * 70)
        for entity_name, entity in self.entities.items():
            print(f"  {entity}")
        
        if self.temporaries:
            print("\nTemporaries:")
            print("-" * 70)
            for temp, value in self.temporaries.items():
                print(f"  {temp} = {value}")
        
        if self.output:
            print("\nOutput:")
            print("-" * 70)
            for line in self.output:
                print(f"  {line}")
        
        print("=" * 70)


# ============================================================================
# INTERPRETER
# ============================================================================

class Interpreter:
    """Interprets and executes three-address code"""
    
    def __init__(self, instructions: List[TACInstruction]):
        """Initialize interpreter"""
        self.instructions = instructions
        self.env = RuntimeEnvironment()
        self.pc = 0  # Program counter
        self.labels: Dict[str, int] = {}  # Label to instruction index mapping
        self.execution_trace: List[str] = []
        self.trace_enabled = False
    
    def enable_trace(self):
        """Enable execution tracing"""
        self.trace_enabled = True
    
    def trace(self, msg: str):
        """Log execution trace"""
        if self.trace_enabled:
            self.execution_trace.append(f"[{self.pc:3d}] {msg}")
    
    def build_label_table(self):
        """Build mapping of labels to instruction indices"""
        for i, instr in enumerate(self.instructions):
            if instr.op == 'label':
                self.labels[instr.result] = i
    
    def execute(self) -> RuntimeEnvironment:
        """Execute the three-address code"""
        self.build_label_table()
        self.pc = 0
        
        print("\n" + "=" * 70)
        print("EXECUTING PROGRAM")
        print("=" * 70)
        
        while self.pc < len(self.instructions):
            instr = self.instructions[self.pc]
            self.execute_instruction(instr)
            self.pc += 1
        
        print("\n✓ Execution completed")
        
        return self.env
    
    def execute_instruction(self, instr: TACInstruction):
        """Execute a single instruction"""
        
        if instr.op == 'label':
            self.trace(f"Label: {instr.result}")
        
        elif instr.op == '=':
            # Assignment: result = arg1
            value = self.get_value(instr.arg1)
            self.set_value(instr.result, value)
            self.trace(f"{instr.result} = {value}")
        
        elif instr.op in ['+', '-', '*', '/']:
            # Arithmetic operation
            left = self.get_value(instr.arg1)
            right = self.get_value(instr.arg2)
            
            if instr.op == '+':
                result = left + right
            elif instr.op == '-':
                result = left - right
            elif instr.op == '*':
                result = left * right
            elif instr.op == '/':
                if right == 0:
                    print(f"Runtime Error: Division by zero at instruction {self.pc}")
                    result = 0
                else:
                    result = left // right
            
            self.set_value(instr.result, result)
            self.trace(f"{instr.result} = {left} {instr.op} {right} = {result}")
        
        elif instr.op in ['==', '!=', '>', '<']:
            # Comparison operation
            left = self.get_value(instr.arg1)
            right = self.get_value(instr.arg2)
            
            if instr.op == '==':
                result = 1 if left == right else 0
            elif instr.op == '!=':
                result = 1 if left != right else 0
            elif instr.op == '>':
                result = 1 if left > right else 0
            elif instr.op == '<':
                result = 1 if left < right else 0
            
            self.set_value(instr.result, result)
            self.trace(f"{instr.result} = {left} {instr.op} {right} = {result}")
        
        elif instr.op == 'goto':
            # Unconditional jump
            target = instr.result
            if target in self.labels:
                self.pc = self.labels[target] - 1  # -1 because pc will increment
                self.trace(f"goto {target} (jump to {self.pc + 1})")
        
        elif instr.op == 'if':
            # Conditional jump: if arg1 goto result
            condition = self.get_value(instr.arg1)
            if condition:
                target = instr.result
                if target in self.labels:
                    self.pc = self.labels[target] - 1
                    self.trace(f"if {condition} goto {target} (taken, jump to {self.pc + 1})")
            else:
                self.trace(f"if {condition} goto {instr.result} (not taken)")
        
        elif instr.op == 'ifFalse':
            # Conditional jump: ifFalse arg1 goto result
            condition = self.get_value(instr.arg1)
            if not condition:
                target = instr.result
                if target in self.labels:
                    self.pc = self.labels[target] - 1
                    self.trace(f"ifFalse {condition} goto {target} (taken, jump to {self.pc + 1})")
            else:
                self.trace(f"ifFalse {condition} goto {instr.result} (not taken)")
        
        elif instr.op == 'move':
            # Move entity: move entity direction amount
            entity_name = instr.arg1
            direction = instr.arg2
            amount = self.get_value(instr.result)
            
            # Get current position
            x = self.env.get_property(entity_name, 'x') or 0
            y = self.env.get_property(entity_name, 'y') or 0
            
            # Update position based on direction
            if direction == 'right':
                x += amount
            elif direction == 'left':
                x -= amount
            elif direction == 'up':
                y += amount
            elif direction == 'down':
                y -= amount
            
            # Set new position
            self.env.set_property(entity_name, 'x', x)
            self.env.set_property(entity_name, 'y', y)
            
            self.trace(f"move {entity_name} {direction} {amount} -> ({x}, {y})")
            print(f"  → {entity_name} moved {direction} by {amount} to position ({x}, {y})")
        
        elif instr.op == 'print':
            # Print message
            message = instr.arg1.strip('"')
            print(f"  → OUTPUT: {message}")
            self.env.add_output(message)
            self.trace(f"print {message}")
    
    def get_value(self, operand: Optional[str]) -> int:
        """Get value of an operand (constant, temporary, or property)"""
        if operand is None:
            return 0
        
        # Check if it's a constant
        try:
            return int(operand)
        except ValueError:
            pass
        
        # Check if it's a property access (entity.property)
        if '.' in operand:
            parts = operand.split('.')
            if len(parts) == 2:
                entity_name, prop_name = parts
                value = self.env.get_property(entity_name, prop_name)
                if value is not None:
                    return value
                else:
                    print(f"Warning: Property {operand} not found, defaulting to 0")
                    return 0
        
        # Check if it's a temporary variable
        value = self.env.get_temp(operand)
        if value is not None:
            return value
        
        # Unknown variable
        print(f"Warning: Variable {operand} not found, defaulting to 0")
        return 0
    
    def set_value(self, target: Optional[str], value: int):
        """Set value of a target (temporary or property)"""
        if target is None:
            return
        
        # Check if it's a property assignment (entity.property)
        if '.' in target:
            parts = target.split('.')
            if len(parts) == 2:
                entity_name, prop_name = parts
                
                # Create entity if it doesn't exist
                if entity_name not in self.env.entities:
                    self.env.create_entity(entity_name, 'unknown')
                
                self.env.set_property(entity_name, prop_name, value)
                return
        
        # Otherwise, it's a temporary variable
        self.env.set_temp(target, value)
    
    def print_trace(self):
        """Print execution trace"""
        if not self.execution_trace:
            print("\nNo execution trace available (tracing not enabled)")
            return
        
        print("\n" + "=" * 70)
        print("EXECUTION TRACE")
        print("=" * 70)
        for line in self.execution_trace:
            print(line)
        print("=" * 70)


# ============================================================================
# COMPLETE COMPILER PIPELINE
# ============================================================================

def compile_and_run(source_code: str, trace: bool = False):
    """Complete compilation and execution pipeline"""
    from lexer import Lexer
    from parser import Parser
    from semantic_analyzer import SemanticAnalyzer
    from code_generator import IntermediateCodeGenerator
    from optimizer import Optimizer
    
    print("=" * 70)
    print("TINYGAME COMPILER - COMPLETE PIPELINE")
    print("=" * 70)
    
    # Phase 1: Lexical Analysis
    print("\n[Phase 1] Lexical Analysis...")
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()
    print(f"✓ Generated {len(tokens)} tokens")
    
    # Phase 2: Syntax Analysis
    print("\n[Phase 2] Syntax Analysis...")
    parser = Parser(tokens)
    ast = parser.parse()
    print("✓ AST built successfully")
    
    # Phase 3: Semantic Analysis
    print("\n[Phase 3] Semantic Analysis...")
    analyzer = SemanticAnalyzer(ast)
    if not analyzer.analyze():
        print("✗ Semantic errors found:")
        for error in analyzer.errors:
            print(f"  {error}")
        return None
    print("✓ No semantic errors")
    
    # Phase 4: Intermediate Code Generation
    print("\n[Phase 4] Intermediate Code Generation...")
    code_gen = IntermediateCodeGenerator(ast, analyzer.symbol_table)
    tac_instructions = code_gen.generate()
    print(f"✓ Generated {len(tac_instructions)} TAC instructions")
    
    # Phase 5: Optimization
    print("\n[Phase 5] Optimization...")
    optimizer = Optimizer(tac_instructions)
    optimized_code = optimizer.optimize()
    print(f"✓ Optimized to {len(optimized_code)} instructions")
    
    # Phase 6: Execution
    print("\n[Phase 6] Code Execution...")
    interpreter = Interpreter(optimized_code)
    if trace:
        interpreter.enable_trace()
    
    runtime_env = interpreter.execute()
    
    # Print results
    runtime_env.print_state()
    
    if trace:
        interpreter.print_trace()
    
    return runtime_env


# ============================================================================
# TESTING THE INTERPRETER
# ============================================================================

if __name__ == "__main__":
    
    # Test Case 1: Basic movement and collision
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
        if hero.y == monster.y {
            set hero.health = hero.health - 10;
            print "Hit by monster!";
        }
    }
    """
    
    # Test Case 2: Score system
    test_code_2 = """
    player hero {
        x = 0;
        y = 0;
        score = 0;
    }
    
    move hero right 10;
    
    if hero.x > 5 {
        set hero.score = hero.score + 100;
        print "Bonus points earned!";
    }
    """
    
    # Test Case 3: Multiple movements
    test_code_3 = """
    player hero {
        x = 5;
        y = 5;
    }
    
    move hero right 3;
    move hero up 2;
    move hero left 1;
    move hero down 1;
    
    print "Final position reached!";
    """
    
    print("\n\n")
    print("#" * 70)
    print("# TEST CASE 1: Collision Detection")
    print("#" * 70)
    compile_and_run(test_code_1)
    
    print("\n\n")
    print("#" * 70)
    print("# TEST CASE 2: Score System")
    print("#" * 70)
    compile_and_run(test_code_2)
    
    print("\n\n")
    print("#" * 70)
    print("# TEST CASE 3: Multiple Movements")
    print("#" * 70)
    compile_and_run(test_code_3, trace=False)
