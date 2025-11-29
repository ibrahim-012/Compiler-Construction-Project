"""
TinyGame Compiler - Optimizer
Performs basic optimizations on three-address code
"""

from code_generator import TACInstruction, IntermediateCodeGenerator
from typing import List, Set, Dict, Optional
from copy import deepcopy

# ============================================================================
# OPTIMIZER CLASS
# ============================================================================

class Optimizer:
    """Performs optimization passes on three-address code"""
    
    def __init__(self, instructions: List[TACInstruction]):
        """Initialize optimizer with TAC instructions"""
        self.instructions = instructions
        self.optimized_instructions: List[TACInstruction] = []
        self.optimization_log: List[str] = []
    
    def log(self, msg: str):
        """Log optimization action"""
        self.optimization_log.append(msg)
    
    def optimize(self) -> List[TACInstruction]:
        """Perform all optimization passes"""
        print("\n" + "=" * 70)
        print("OPTIMIZATION PASSES")
        print("=" * 70)
        
        # Start with original instructions
        current_instructions = deepcopy(self.instructions)
        original_count = len(current_instructions)
        
        print(f"\nOriginal instruction count: {original_count}")
        
        # Pass 1: Constant Folding
        print("\n[Pass 1] Constant Folding...")
        current_instructions = self.constant_folding(current_instructions)
        print(f"  ✓ Completed. {len(self.optimization_log)} optimizations applied")
        
        # Pass 2: Dead Code Elimination
        print("\n[Pass 2] Dead Code Elimination...")
        current_instructions = self.dead_code_elimination(current_instructions)
        print(f"  ✓ Completed")
        
        # Pass 3: Copy Propagation
        print("\n[Pass 3] Copy Propagation...")
        current_instructions = self.copy_propagation(current_instructions)
        print(f"  ✓ Completed")
        
        # Pass 4: Algebraic Simplification
        print("\n[Pass 4] Algebraic Simplification...")
        current_instructions = self.algebraic_simplification(current_instructions)
        print(f"  ✓ Completed")
        
        final_count = len(current_instructions)
        reduction = original_count - final_count
        
        print("\n" + "=" * 70)
        print(f"Optimization complete!")
        print(f"Instructions reduced: {original_count} → {final_count} (-{reduction})")
        print("=" * 70)
        
        self.optimized_instructions = current_instructions
        return current_instructions
    
    def constant_folding(self, instructions: List[TACInstruction]) -> List[TACInstruction]:
        """
        Constant Folding: Evaluate constant expressions at compile time
        Example: t0 = 5 + 3  →  t0 = 8
        """
        optimized = []
        
        for instr in instructions:
            # Check if this is an arithmetic operation with constant operands
            if instr.op in ['+', '-', '*', '/'] and instr.arg1 and instr.arg2:
                # Check if both arguments are integer constants
                if self.is_constant(instr.arg1) and self.is_constant(instr.arg2):
                    val1 = int(instr.arg1)
                    val2 = int(instr.arg2)
                    
                    # Compute result
                    result_val = None
                    if instr.op == '+':
                        result_val = val1 + val2
                    elif instr.op == '-':
                        result_val = val1 - val2
                    elif instr.op == '*':
                        result_val = val1 * val2
                    elif instr.op == '/':
                        if val2 != 0:
                            result_val = val1 // val2
                    
                    if result_val is not None:
                        # Replace with constant assignment
                        new_instr = TACInstruction('=', str(result_val), None, instr.result)
                        optimized.append(new_instr)
                        self.log(f"Constant folding: {instr.arg1} {instr.op} {instr.arg2} = {result_val}")
                        continue
            
            # No optimization applied, keep original
            optimized.append(instr)
        
        return optimized
    
    def dead_code_elimination(self, instructions: List[TACInstruction]) -> List[TACInstruction]:
        """
        Dead Code Elimination: Remove instructions whose results are never used
        """
        # First pass: identify all variables that are used
        used_vars: Set[str] = set()
        
        for instr in instructions:
            # Mark arguments as used
            if instr.arg1 and not self.is_constant(instr.arg1):
                used_vars.add(instr.arg1)
            if instr.arg2 and not self.is_constant(instr.arg2):
                used_vars.add(instr.arg2)
            
            # Special handling for control flow and I/O
            if instr.op in ['if', 'ifFalse', 'goto', 'print', 'move']:
                if instr.arg1:
                    used_vars.add(instr.arg1)
                if instr.result:
                    used_vars.add(instr.result)
        
        # Second pass: keep only instructions that:
        # 1. Produce a result that is used, OR
        # 2. Have side effects (labels, jumps, prints, moves, property assignments)
        optimized = []
        
        for instr in instructions:
            keep = False
            
            # Always keep labels, jumps, prints, and moves
            if instr.op in ['label', 'goto', 'if', 'ifFalse', 'print', 'move']:
                keep = True
            
            # Always keep property assignments (side effects)
            elif instr.op == '=' and instr.result and '.' in instr.result:
                keep = True
            
            # Keep if result is used later
            elif instr.result and instr.result in used_vars:
                keep = True
            
            if keep:
                optimized.append(instr)
            else:
                self.log(f"Dead code eliminated: {instr}")
        
        return optimized
    
    def copy_propagation(self, instructions: List[TACInstruction]) -> List[TACInstruction]:
        """
        Copy Propagation: Replace uses of copied variables with their source
        Example: 
            t0 = x
            t1 = t0 + 5
        Becomes:
            t0 = x
            t1 = x + 5
        """
        optimized = []
        copy_map: Dict[str, str] = {}  # Maps variables to what they copy
        
        for instr in instructions:
            # Update copy map for simple assignments
            if instr.op == '=' and instr.arg1 and not self.is_constant(instr.arg1) and instr.result:
                # Check if this is a simple copy (not a property)
                if '.' not in instr.arg1 and not self.is_temp(instr.arg1):
                    copy_map[instr.result] = instr.arg1
            
            # Propagate copies in current instruction
            new_instr = deepcopy(instr)
            
            if new_instr.arg1 and new_instr.arg1 in copy_map:
                original = new_instr.arg1
                new_instr.arg1 = copy_map[new_instr.arg1]
                self.log(f"Copy propagation: {original} → {new_instr.arg1}")
            
            if new_instr.arg2 and new_instr.arg2 in copy_map:
                original = new_instr.arg2
                new_instr.arg2 = copy_map[new_instr.arg2]
                self.log(f"Copy propagation: {original} → {new_instr.arg2}")
            
            optimized.append(new_instr)
            
            # Invalidate copy map entries when variables are redefined
            if new_instr.result:
                # Remove this variable from copy map
                if new_instr.result in copy_map:
                    del copy_map[new_instr.result]
                
                # Remove any copies of this variable
                to_remove = [k for k, v in copy_map.items() if v == new_instr.result]
                for k in to_remove:
                    del copy_map[k]
        
        return optimized
    
    def algebraic_simplification(self, instructions: List[TACInstruction]) -> List[TACInstruction]:
        """
        Algebraic Simplification: Simplify expressions using algebraic identities
        Examples:
            x + 0 → x
            x - 0 → x
            x * 1 → x
            x * 0 → 0
            x / 1 → x
        """
        optimized = []
        
        for instr in instructions:
            simplified = False
            
            if instr.op in ['+', '-', '*', '/'] and instr.arg1 and instr.arg2 and instr.result:
                arg1 = instr.arg1
                arg2 = instr.arg2
                
                # x + 0 = x  or  0 + x = x
                if instr.op == '+':
                    if arg2 == '0':
                        new_instr = TACInstruction('=', arg1, None, instr.result)
                        optimized.append(new_instr)
                        self.log(f"Algebraic simplification: {arg1} + 0 → {arg1}")
                        simplified = True
                    elif arg1 == '0':
                        new_instr = TACInstruction('=', arg2, None, instr.result)
                        optimized.append(new_instr)
                        self.log(f"Algebraic simplification: 0 + {arg2} → {arg2}")
                        simplified = True
                
                # x - 0 = x
                elif instr.op == '-' and arg2 == '0':
                    new_instr = TACInstruction('=', arg1, None, instr.result)
                    optimized.append(new_instr)
                    self.log(f"Algebraic simplification: {arg1} - 0 → {arg1}")
                    simplified = True
                
                # x * 1 = x  or  1 * x = x
                elif instr.op == '*':
                    if arg2 == '1':
                        new_instr = TACInstruction('=', arg1, None, instr.result)
                        optimized.append(new_instr)
                        self.log(f"Algebraic simplification: {arg1} * 1 → {arg1}")
                        simplified = True
                    elif arg1 == '1':
                        new_instr = TACInstruction('=', arg2, None, instr.result)
                        optimized.append(new_instr)
                        self.log(f"Algebraic simplification: 1 * {arg2} → {arg2}")
                        simplified = True
                    # x * 0 = 0  or  0 * x = 0
                    elif arg2 == '0' or arg1 == '0':
                        new_instr = TACInstruction('=', '0', None, instr.result)
                        optimized.append(new_instr)
                        self.log(f"Algebraic simplification: {arg1} * {arg2} → 0")
                        simplified = True
                
                # x / 1 = x
                elif instr.op == '/' and arg2 == '1':
                    new_instr = TACInstruction('=', arg1, None, instr.result)
                    optimized.append(new_instr)
                    self.log(f"Algebraic simplification: {arg1} / 1 → {arg1}")
                    simplified = True
            
            if not simplified:
                optimized.append(instr)
        
        return optimized
    
    def is_constant(self, value: str) -> bool:
        """Check if a value is a constant (integer literal)"""
        if not value:
            return False
        try:
            int(value)
            return True
        except ValueError:
            return False
    
    def is_temp(self, value: str) -> bool:
        """Check if a value is a temporary variable"""
        return value.startswith('t') and value[1:].isdigit()
    
    def print_optimizations(self):
        """Print optimization log"""
        print("\n" + "=" * 70)
        print("OPTIMIZATION LOG")
        print("=" * 70)
        
        if self.optimization_log:
            for i, log in enumerate(self.optimization_log, 1):
                print(f"{i:3d}. {log}")
        else:
            print("No optimizations applied")
        
        print("=" * 70)
    
    def print_comparison(self):
        """Print before/after comparison"""
        print("\n" + "=" * 70)
        print("BEFORE vs AFTER OPTIMIZATION")
        print("=" * 70)
        
        print("\nORIGINAL CODE:")
        print("-" * 70)
        for i, instr in enumerate(self.instructions):
            print(f"{i:3d}  {instr}")
        
        print("\n\nOPTIMIZED CODE:")
        print("-" * 70)
        for i, instr in enumerate(self.optimized_instructions):
            print(f"{i:3d}  {instr}")
        
        print("\n" + "=" * 70)


# ============================================================================
# TESTING THE OPTIMIZER
# ============================================================================

if __name__ == "__main__":
    from lexer import Lexer
    from parser import Parser
    from semantic_analyzer import SemanticAnalyzer
    
    # Test code with opportunities for optimization
    test_code = """
    player hero {
        x = 5 + 3;
        y = 10 * 1;
        health = 100 - 0;
        score = 0 * 999;
    }
    
    enemy monster {
        x = 2 + 2;
        y = 8 / 1;
    }
    
    move hero right 3;
    set hero.score = hero.score + 0;
    
    if hero.x == monster.x {
        set hero.health = hero.health - 5;
        print "Hit!";
    }
    """
    
    print("=" * 70)
    print("TINYGAME OPTIMIZER TEST")
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
        print("✗ Semantic errors found")
        for error in analyzer.errors:
            print(f"  {error}")
        exit(1)
    
    # Intermediate code generation
    print("\n[4] Intermediate Code Generation...")
    code_gen = IntermediateCodeGenerator(ast, analyzer.symbol_table)
    tac_instructions = code_gen.generate()
    print(f"✓ Generated {len(tac_instructions)} TAC instructions")
    
    # Optimization
    print("\n[5] Optimization...")
    optimizer = Optimizer(tac_instructions)
    optimized_code = optimizer.optimize()
    
    # Print results
    optimizer.print_optimizations()
    optimizer.print_comparison()
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"Original instructions:  {len(tac_instructions)}")
    print(f"Optimized instructions: {len(optimized_code)}")
    print(f"Reduction:              {len(tac_instructions) - len(optimized_code)} instructions")
    print(f"Optimizations applied:  {len(optimizer.optimization_log)}")
    print("=" * 70)
