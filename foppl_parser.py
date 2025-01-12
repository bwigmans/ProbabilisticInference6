import re
from ast_nodes import Constant, Variable, Let, Sample, If, For, Observe

# Lexer: Tokenizing the input FOPPL program
class FOPPLLexer:
    """Lexer to tokenize the input FOPPL string."""
    def __init__(self, program):
        self.program = program
        self.tokens = []
        self.current_token_index = 0
        self.tokenize()

    def tokenize(self):
        token_patterns = [
            ('KEYWORD_LET', r'\blet\b'),  # Matches the 'let' keyword
            ('KEYWORD_SAMPLE', r'\bsample\b'),  # Matches the 'sample' keyword
            ('KEYWORD_OBSERVE', r'\bobserve\b'),  # Matches the 'observe' keyword
            ('KEYWORD_IF', r'\bif\b'),  # Matches the 'if' keyword
            ('KEYWORD_FOR', r'\bfor\b'),  # Matches the 'for' keyword
            ('NUMBER', r'\d+(\.\d+)?'),  # Matches numbers (integers and floats)
            ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z_0-9]*'),  # Matches identifiers (variable names)
            #TODO maybe add boolean
            ('SYMBOL_LPAREN', r'\('),  # Matches opening parenthesis '('
            ('SYMBOL_RPAREN', r'\)'),  # Matches closing parenthesis ')'
            ('SYMBOL_LBRACKET', r'\['),  # Matches opening square bracket '['
            ('SYMBOL_RBRACKET', r'\]'),  # Matches closing square bracket ']'
            ('SYMBOL_COMMA', r','),  # Matches commas ','
            ('SYMBOL_ASSIGN', r'='),  # Matches equals sign '='
            ('SYMBOL_PLUS', r'\+'),  # Matches plus sign '+'
            ('WHITESPACE', r'\s+'),  # Matches whitespaces (to be ignored)
            ('NEWLINE', r'\n'),  # Matches newlines
            ('COMMENT', r';[^\n]*'),  # Matches comments
        ]
        regex_parts = [f'(?P<{name}>{pattern})' for name, pattern in token_patterns]
        regex = '|'.join(regex_parts)
        for match in re.finditer(regex, self.program):
            kind = match.lastgroup
            value = match.group()
            if kind not in ['WHITESPACE', 'COMMENT']:  # Ignore whitespace and comments
                self.tokens.append((kind, value))
        self.tokens.append(('EOF', ''))  # Add EOF token to indicate the end

    def get_next_token(self):
        """Fetch the next token."""
        if self.current_token_index < len(self.tokens):
            token = self.tokens[self.current_token_index]
            self.current_token_index += 1
            return token
        return ('EOF', '')  # Return EOF token at the end

    def peek_next_token(self):
        """Peek the next token without consuming it."""
        if self.current_token_index < len(self.tokens):
            return self.tokens[self.current_token_index]
        return ('EOF', '')  # Return EOF token when there are no more tokens


class FOPPLParser:
    """Parser to construct the AST from the FOPPL token stream."""
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = None
        self.advance()

    def advance(self):
        """Move to the next token."""
        self.current_token = self.lexer.get_next_token()

    def parse(self):
        """Start parsing the program."""
        return self.program()

    def program(self):
        """Parse the entire program consisting of statements."""
        statements = []
        while self.current_token[0] != 'EOF':
            statements.append(self.statement())
        return statements

    def statement(self):
        """Parse a statement, which can be let, sample, if, etc."""
        if self.current_token[0] == 'KEYWORD_LET':
            return self.let_statement()
        elif self.current_token[0] == 'KEYWORD_SAMPLE':
            return self.sample_statement()
        elif self.current_token[0] == 'KEYWORD_OBSERVE':
            return self.observe_statement()
        elif self.current_token[0] == 'KEYWORD_IF':
            return self.if_statement()
        elif self.current_token[0] == 'KEYWORD_FOR':
            return self.for_statement()
        elif self.current_token[0] == 'SYMBOL_LPAREN':  # Handle sub-expressions in parentheses
            return self.parenthesized_expression()
        else:
            raise SyntaxError(f"Unexpected token: {self.current_token}")

    def let_statement(self):
        """
        Parse a 'let' statement of the form (let [var value] body).
        Also supports multiple (var value) pairs in a single let statement.
        """
        self.advance()  # Skip 'let'
        if self.current_token[0] != 'SYMBOL_LBRACKET':
            raise SyntaxError(
                f"Expected '[', got {self.current_token} in let statement."
            )
        self.advance()  # Skip '['

        pairs = []
        # Collect (var, value_expr) pairs until we see ']'
        while self.current_token[0] != 'SYMBOL_RBRACKET':
            # Parse var name
            if self.current_token[0] != 'IDENTIFIER':
                raise SyntaxError(
                    f"Expected identifier in let, got {self.current_token}"
                )
            var_name = self.current_token[1]
            self.advance()  # Skip the identifier

            # Parse value expression
            saved_token = self.current_token  # Remember where we are
            value_expr = self.expression()    # Let `expression()` produce an AST node
                                              # but it won't call `self.advance()` if single-token

            # If `expression()` did NOT advance (e.g., for a single number or identifier),
            # then consume one token manually.
            if self.current_token == saved_token:
                self.advance()  # Force consumption of that single token

            # Add (var, value_expr) to our list of pairs
            pairs.append((var_name, value_expr))

        # At this point, we must see 'SYMBOL_RBRACKET'
        if self.current_token[0] != 'SYMBOL_RBRACKET':
            raise SyntaxError(
                f"Expected ']', got {self.current_token} in let statement."
            )
        self.advance()  # Skip ']'

        # Parse the body (the statement after [pairs])
        body_expr = self.statement()

        # Build nested Let structures: Let(var1, val1, Let(var2, val2, ... body))
        ast = body_expr
        for var, val in reversed(pairs):
            ast = Let(var, val, ast)

        return ast

    def sample_statement(self):
        """Parse a 'sample' statement of the form (sample var (distribution args))."""
        self.advance()  # Skip 'sample'
        var_name = self.current_token[1]
        self.advance()  # Skip var name
        self.advance()  # Skip '('
        distribution_expr = self.distribution_expression()
        self.advance()  # Skip ')'
        return Sample(var_name, distribution_expr)

    def if_statement(self):
        """Parse an 'if' statement of the form (if condition then else)."""
        self.advance()  # Skip '('
        self.advance()  # Skip 'if'
        condition = self.expression()
        then_branch = self.statement()
        self.advance()  # Skip 'else' keyword
        else_branch = self.statement()
        return If(condition, then_branch, else_branch)

    #TODO test this with a simple test case
    def for_statement(self):
        """Parse a 'for' statement of the form (for [var range] body)."""
        # self.advance()  # Skip '('
        self.advance()  # Skip 'for'
        self.advance()  # Skip '['
        loop_var = self.current_token[1]
        self.advance()  # Skip loop variable
        range_n = self.expression()
        self.advance()  # Skip range
        self.advance()  # Skip ']'
        body_expr = self.statement()
        return For(loop_var, range_n, body_expr)
    
    def observe_statement(self):
        """Parse an 'observe' statement of the form (observe distribution variable)."""
        self.advance()  # Skip 'observe'
        self.advance()  # Skip '('
        distribution_expr = self.distribution_expression()
        self.advance()  # Skip ')'
        observed_var = self.expression()
        self.advance()  # Skip observed_var
        return Observe(distribution_expr, observed_var)


    def expression(self):
        """Parse an expression (either a constant or a variable)."""
        if self.current_token[0] == 'NUMBER':
            return Constant(float(self.current_token[1]))  # Return constant value
        elif self.current_token[0] == 'IDENTIFIER':
            return Variable(self.current_token[1])  # Return variable name
        elif self.current_token[0] == 'SYMBOL_LPAREN':  # Handle parenthesized expressions
            return self.parenthesized_expression()  # Call to handle parenthesized sub-expressions
        else:
            raise SyntaxError(f"Unexpected token in expression: {self.current_token}")

    def distribution_expression(self):
        """Parse a distribution expression like (normal x 1)."""
        dist_name = self.current_token[1]
        self.advance()  # Skip distribution name
        arg1 = self.expression()
        self.advance()  # Skip arg1
        arg2 = self.expression()
        self.advance() # Skip arg2
        return (dist_name, arg1, arg2)  # Return a tuple representing the distribution

    def parenthesized_expression(self):
        """Handle expressions inside parentheses."""
        self.advance()  # Skip '('
        expr = self.statement()  # Parse the statement inside the parentheses
        if self.current_token[0] != 'SYMBOL_RPAREN':
            raise SyntaxError(f"Expected ')', found: {self.current_token}")
        self.advance()  # Skip ')'
        return expr


# Example Usage:
if __name__ == "__main__":
    # A simple FOPPL program (Lisp-style syntax)
    code = "(for [x 5] (sample x (normal 0 1)))"
    
    # Step 1: Lexical analysis (tokenization)
    lexer = FOPPLLexer(code)
    
    # Step 2: Parsing the tokens into an AST
    parser = FOPPLParser(lexer)
    ast = parser.parse()

    # Step 3: Printing the AST
    print(ast)

    # print("=====TEST=====")
    # code = "(let [a 2] (sample x (normal 0 1)))"
    # lexer = FOPPLLexer(code)
    # parser = FOPPLParser(lexer)
    # ast = parser.parse()
    # print(ast, "\n")

    # print("=====TEST=====")
    # code = "(let [x (sample x (normal 0 1))] (observe (normal x 1) 3))"
    # lexer = FOPPLLexer(code)
    # parser = FOPPLParser(lexer)
    # ast = parser.parse()
    # print(ast, "\n")

    # print("=====TEST=====")
    # code = "(let [x 2 y 3 z 4] (sample x (normal 0 1)))"
    # lexer = FOPPLLexer(code)
    # parser = FOPPLParser(lexer)
    # ast = parser.parse()
    # print(ast, "\n")
    
