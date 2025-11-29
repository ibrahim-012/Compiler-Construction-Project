## Run a program
python main.py examples/test1.tg

## Show all compilation phases
python main.py examples/test1.tg --all

## Show optimized code
python main.py examples/test2.tg --optimized

## Execute with trace
python main.py examples/test3.tg --trace

## Interactive mode
python main.py --interactive

## Create example files
python main.py --create-examples

## Show help
python main.py --help


## Interactive Mode Example
\>>> player hero {  
\>>> x = 0;  
\>>> y = 0;  
\>>> }  
\>>> move hero right 5;  
\>>> print "Done!";  
\>>> RUN  

Executing code...  
  → hero moved right by 5 to position (5, 0)  
  → OUTPUT: Done!  

✓ Execution completed  


## Key Features

Multiple compilation modes  
Clean CLI interface  
Error handling and reporting  
Interactive REPL mode  
Example file generation  
Help system  
Beautiful output formatting  
Progress tracking  

## Complete File Structure
```
tinygame-compiler/  
├── main.py                  # Main CLI  
├── lexer.py                 # Part 1  
├── parser.py                # Part 2  
├── semantic_analyzer.py     # Part 3  
├── code_generator.py        # Part 4  
├── optimizer.py             # Part 5  
├── interpreter.py           # Part 6  
└── examples/  
    ├── test1.tg  
    ├── test2.tg  
    └── test3.tg
```