import ast
import re
from typing import Optional
from langchain.tools import tool
from langchain_core.callbacks import CallbackManagerForToolRun

@tool
def ast_parser(code: str) -> str:
    \"\"\"Parse Python code using AST to detect syntax errors and analyze structure.
    
    Args:
        code: Python code string
        
    Returns:
        Analysis or error message
    \"\"\"
    try:
        tree = ast.parse(code)
        # Extract variable names, function defs
        visitor = VariableVisitor()
        visitor.visit(tree)
        vars_used = set(visitor.variables)
        funcs = [node.name for node in visitor.funcs]
        
        analysis = f"Syntax OK. Variables used: {list(vars_used)}. Functions: {funcs}"
        return analysis
    except SyntaxError as e:
        return f"SyntaxError: {str(e)} at line {e.lineno}"
    except Exception as e:
        return f"Analysis error: {str(e)}"

class VariableVisitor(ast.NodeVisitor):
    def __init__(self):
        self.variables = set()
        self.funcs = []

    def visit_Name(self, node):
        self.variables.add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.funcs.append(node.name)
        self.generic_visit(node)

@tool
def static_analyzer(code: str) -> str:
    \"\"\"Perform static analysis for common Python errors (indentation, common patterns).
    
    Args:
        code: Python code string
        
    Returns:
        List of potential issues
    \"\"\"
    issues = []
    
    # Indentation check
    lines = code.split('\\n')
    for i, line in enumerate(lines):
        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces % 4 != 0 and line.strip():
            issues.append(f"Line {i+1}: Non-standard indentation ({leading_spaces} spaces)")
    
    # Undefined var patterns
    undef_pattern = r'\\b(?!if|for|while|def|class|import|from|return|print|len|str|int|list|dict|tuple|set|range|open)\\\\w+\\\\b'
    words = re.findall(undef_pattern, code)
    potential_undef = [w for w in set(words) if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', w)]
    if potential_undef:
        issues.append(f"Potential undefined: {potential_undef[:5]}")
    
    # Common errors
    if '==' in code and '=' in code and "print(" not in code:
        issues.append("Check == vs = confusion")
    
    return "; ".join(issues[:5]) or "No obvious static issues found."

@tool
def error_classifier(traceback: str) -> str:
    \"\"\"Classify Python error type from traceback.
    
    Args:
        traceback: Error traceback string
        
    Returns:
        Detected error type
    \"\"\"
    error_types = {
        'NameError': r"NameError: name '.*?' is not defined",
        'TypeError': r"TypeError: .*?",
        'IndexError': r"IndexError: (list|tuple|str) index out of range",
        'KeyError': r"KeyError: '.*?'",
        'SyntaxError': r"SyntaxError: ",
        'IndentationError': r"IndentationError: ",
        'ImportError': r"ImportError: ",
        'AttributeError': r"AttributeError: ",
    }
    
    for err_type, pattern in error_types.items():
        if re.search(pattern, traceback, re.IGNORECASE):
            return err_type
    
    return "Unknown"

