"""
TinyGame Compiler - Main Program
Command-line interface for the complete compiler
"""

import sys
import os
from lexer import Lexer, TokenType
from parser import Parser, ASTPrinter
from semantic_analyzer import SemanticAnalyzer
from code_generator import IntermediateCodeGenerator
from optimizer import Optimizer
from interpreter import Interpreter, compile_and_run

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

class TinyGameCompiler:
    """Main compiler class with CLI"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.source_code = None
        self.filename = None
    
    def print_banner(self):
        """Print compiler banner"""
        print("=" * 70)
        print(" _____ _             ____                      ")
        print("|_   _(_)_ __  _   _/ ___| __ _ _ __ ___   ___ ")
        print("  | | | | '_ \\| | | | |  _ / _` | '_ ` _ \\ / _ \\")
        print("  | | | | | | | |_| | |_| | (_| | | | | | |  __/")
        print("  |_| |_|_| |_|\\__, |\\____|\\__,_|_| |_| |_|\\___|")
        print("               |___/                            ")
        print("")
        print(f"  TinyGame Compiler v{self.version}")
        print("  A Mini Language Compiler for CS4031")
        print("=" * 70)
    
    def print_help(self):
        """Print help message"""
        print("\nUsage:")
        print("  python main.py <filename>              - Compile and run a file")
        print("  python main.py <filename> --tokens     - Show tokens only")
        print("  python main.py <filename> --ast        - Show AST only")
        print("  python main.py <filename> --tac        - Show TAC only")
        print("  python main.py <filename> --optimized  - Show optimized TAC")
        print("  python main.py <filename> --trace      - Execute with trace")
        print("  python main.py <filename> --all        - Show all phases")
        print("  python main.py --interactive           - Interactive mode")
        print("  python main.py --help                  - Show this help")
        print("\nExamples:")
        print("  python main.py examples/test1.tg")
        print("  python main.py examples/test2.tg --trace")
        print("  python main.py examples/test3.tg --all")
    
    def read_file(self, filename: str) -> str:
        """Read source code from file"""
        try:
            with open(filename, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    
    def compile_file(self, filename: str, mode: str = 'run'):
        """Compile a source file with specified mode"""
        self.filename = filename
        self.source_code = self.read_file(filename)
        
        print(f"\nCompiling: {filename}")
        print("=" * 70)
        
        try:
            if mode == 'tokens':
                self.show_tokens_only()
            elif mode == 'ast':
                self.show_ast_only()
            elif mode == 'tac':
                self.show_tac_only()
            elif mode == 'optimized':
                self.show_optimized_only()
            elif mode == 'trace':
                self.run_with_trace()
            elif mode == 'all':
                self.show_all_phases()
            else:  # default: run
                self.run_program()
        
        except Exception as e:
            print(f"\n❌ Compilation failed with error:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def show_tokens_only(self):
        """Show lexical analysis only"""
        print("\n[LEXICAL ANALYSIS]")
        print("-" * 70)
        
        lexer = Lexer(self.source_code)
        tokens = lexer.tokenize()
        
        print(f"\nTotal tokens: {len(tokens)}\n")
        for i, token in enumerate(tokens):
            if token.type != TokenType.EOF:
                print(f"{i:4d}. {token}")
        
        print("\n✓ Lexical analysis completed")
    
    def show_ast_only(self):
        """Show syntax analysis only"""
        print("\n[LEXICAL ANALYSIS]")
        lexer = Lexer(self.source_code)
        tokens = lexer.tokenize()
        print(f"✓ Generated {len(tokens)} tokens")
        
        print("\n[SYNTAX ANALYSIS]")
        print("-" * 70)
        
        parser = Parser(tokens)
        ast = parser.parse()
        
        print("\nAbstract Syntax Tree:")
        print(ASTPrinter.print_ast(ast))
        
        print("✓ Syntax analysis completed")
    
    def show_tac_only(self):
        """Show intermediate code generation only"""
        print("\n[PHASES 1-4: Through Intermediate Code Generation]")
        print("-" * 70)
        
        # Lexical
        lexer = Lexer(self.source_code)
        tokens = lexer.tokenize()
        print(f"✓ Lexical: {len(tokens)} tokens")
        
        # Syntax
        parser = Parser(tokens)
        ast = parser.parse()
        print("✓ Syntax: AST built")
        
        # Semantic
        analyzer = SemanticAnalyzer(ast)
        if not analyzer.analyze():
            print("\n❌ Semantic errors found:")
            for error in analyzer.errors:
                print(f"   {error}")
            return
        print("✓ Semantic: No errors")
        
        # Intermediate code
        code_gen = IntermediateCodeGenerator(ast, analyzer.symbol_table)
        tac = code_gen.generate()
        print(f"✓ IR Generation: {len(tac)} instructions")
        
        code_gen.print_code()
    
    def show_optimized_only(self):
        """Show optimization results only"""
        print("\n[PHASES 1-5: Through Optimization]")
        print("-" * 70)
        
        # Lexical
        lexer = Lexer(self.source_code)
        tokens = lexer.tokenize()
        print(f"✓ Lexical: {len(tokens)} tokens")
        
        # Syntax
        parser = Parser(tokens)
        ast = parser.parse()
        print("✓ Syntax: AST built")
        
        # Semantic
        analyzer = SemanticAnalyzer(ast)
        if not analyzer.analyze():
            print("\n❌ Semantic errors found:")
            for error in analyzer.errors:
                print(f"   {error}")
            return
        print("✓ Semantic: No errors")
        
        # Intermediate code
        code_gen = IntermediateCodeGenerator(ast, analyzer.symbol_table)
        tac = code_gen.generate()
        print(f"✓ IR Generation: {len(tac)} instructions")
        
        # Optimization
        optimizer = Optimizer(tac)
        optimized = optimizer.optimize()
        
        optimizer.print_comparison()
        optimizer.print_optimizations()
    
    def run_with_trace(self):
        """Run program with execution trace"""
        compile_and_run(self.source_code, trace=True)
    
    def run_program(self):
        """Run program normally"""
        compile_and_run(self.source_code, trace=False)
    
    def show_all_phases(self):
        """Show detailed output for all phases"""
        print("\n" + "=" * 70)
        print("COMPLETE COMPILATION PROCESS - ALL PHASES")
        print("=" * 70)
        
        # Phase 1: Lexical
        print("\n" + "=" * 70)
        print("PHASE 1: LEXICAL ANALYSIS")
        print("=" * 70)
        lexer = Lexer(self.source_code)
        tokens = lexer.tokenize()
        print(f"\n✓ Generated {len(tokens)} tokens")
        print("\nFirst 20 tokens:")
        for i, token in enumerate(tokens[:20]):
            print(f"  {i:3d}. {token}")
        if len(tokens) > 20:
            print(f"  ... and {len(tokens) - 20} more tokens")
        
        # Phase 2: Syntax
        print("\n" + "=" * 70)
        print("PHASE 2: SYNTAX ANALYSIS")
        print("=" * 70)
        parser = Parser(tokens)
        ast = parser.parse()
        print("\n✓ AST built successfully")
        print("\nAbstract Syntax Tree:")
        print(ASTPrinter.print_ast(ast))
        
        # Phase 3: Semantic
        print("\n" + "=" * 70)
        print("PHASE 3: SEMANTIC ANALYSIS")
        print("=" * 70)
        analyzer = SemanticAnalyzer(ast)
        if not analyzer.analyze():
            print("\n❌ Semantic errors found:")
            for error in analyzer.errors:
                print(f"   {error}")
            return
        print("\n✓ No semantic errors")
        print(analyzer.symbol_table)
        
        # Phase 4: Intermediate Code
        print("\n" + "=" * 70)
        print("PHASE 4: INTERMEDIATE CODE GENERATION")
        print("=" * 70)
        code_gen = IntermediateCodeGenerator(ast, analyzer.symbol_table)
        tac = code_gen.generate()
        print(f"\n✓ Generated {len(tac)} TAC instructions")
        code_gen.print_code()
        
        # Phase 5: Optimization
        print("\n" + "=" * 70)
        print("PHASE 5: OPTIMIZATION")
        print("=" * 70)
        optimizer = Optimizer(tac)
        optimized = optimizer.optimize()
        optimizer.print_optimizations()
        
        # Phase 6: Execution
        print("\n" + "=" * 70)
        print("PHASE 6: CODE EXECUTION")
        print("=" * 70)
        interpreter = Interpreter(optimized)
        runtime_env = interpreter.execute()
        runtime_env.print_state()
    
    def interactive_mode(self):
        """Interactive REPL mode"""
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        print("\nEnter TinyGame code line by line.")
        print("Type 'RUN' to execute, 'CLEAR' to reset, 'EXIT' to quit.")
        print("-" * 70)
        
        lines = []
        
        while True:
            try:
                line = input(">>> ")
                
                if line.strip().upper() == 'EXIT':
                    print("\nGoodbye!")
                    break
                
                elif line.strip().upper() == 'CLEAR':
                    lines = []
                    print("Code cleared.")
                    continue
                
                elif line.strip().upper() == 'RUN':
                    if not lines:
                        print("No code to run. Enter some code first.")
                        continue
                    
                    source = '\n'.join(lines)
                    print("\n" + "-" * 70)
                    print("Executing code:")
                    print("-" * 70)
                    print(source)
                    print("-" * 70)
                    
                    try:
                        compile_and_run(source)
                    except Exception as e:
                        print(f"\n❌ Error: {e}")
                    
                    print("\n" + "-" * 70)
                    print("Enter more code or type RUN to execute again.")
                    continue
                
                else:
                    lines.append(line)
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break


# ============================================================================
# EXAMPLE FILE GENERATOR
# ============================================================================

def create_example_files():
    """Create example TinyGame files"""
    
    # Create examples directory
    if not os.path.exists('examples'):
        os.makedirs('examples')
    
    # Example 1: Collision detection
    example1 = """player hero {
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
    
    # Example 2: Score system
    example2 = """player hero {
    x = 0;
    y = 0;
    score = 0;
}

move hero right 10;

if hero.x > 5 {
    set hero.score = hero.score + 100;
    print "Bonus points earned!";
}

move hero up 3;
print "Game over!";
"""
    
    # Example 3: Multiple entities
    example3 = """player hero {
    x = 0;
    y = 0;
}

enemy goblin {
    x = 3;
    y = 3;
}

enemy dragon {
    x = 8;
    y = 8;
}

move hero right 3;
move hero up 3;

if hero.x == goblin.x {
    print "Found goblin!";
}

move hero right 5;
move hero up 5;

if hero.x == dragon.x {
    print "Found dragon!";
}
"""
    
    # Write files
    with open('examples/test1.tg', 'w') as f:
        f.write(example1)
    
    with open('examples/test2.tg', 'w') as f:
        f.write(example2)
    
    with open('examples/test3.tg', 'w') as f:
        f.write(example3)
    
    print("✓ Created example files in 'examples/' directory")
    print("  - examples/test1.tg (Collision detection)")
    print("  - examples/test2.tg (Score system)")
    print("  - examples/test3.tg (Multiple entities)")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    compiler = TinyGameCompiler()
    compiler.print_banner()
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        compiler.print_help()
        print("\nℹ️  No file specified. Use --help for usage information.")
        
        # Ask if user wants to create examples
        response = input("\nWould you like to create example files? (y/n): ")
        if response.lower() == 'y':
            create_example_files()
            print("\nYou can now run:")
            print("  python main.py examples/test1.tg")
        
        sys.exit(0)
    
    # Check for flags
    filename = sys.argv[1]
    
    if filename == '--help':
        compiler.print_help()
        sys.exit(0)
    
    if filename == '--interactive':
        compiler.interactive_mode()
        sys.exit(0)
    
    if filename == '--create-examples':
        create_example_files()
        sys.exit(0)
    
    # Determine mode
    mode = 'run'
    if len(sys.argv) > 2:
        flag = sys.argv[2]
        if flag == '--tokens':
            mode = 'tokens'
        elif flag == '--ast':
            mode = 'ast'
        elif flag == '--tac':
            mode = 'tac'
        elif flag == '--optimized':
            mode = 'optimized'
        elif flag == '--trace':
            mode = 'trace'
        elif flag == '--all':
            mode = 'all'
    
    # Compile and run
    compiler.compile_file(filename, mode)


if __name__ == "__main__":
    main()