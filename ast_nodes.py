class Node:
    """Base class for all AST nodes."""
    pass

class Constant(Node):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Constant({self.value})"

class Variable(Node):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Variable({self.name})"

class Let(Node):
    """
    A let form: (let [v e1] e2)
    """
    def __init__(self, var_name, e1, e2):
        self.var_name = var_name
        self.e1 = e1
        self.e2 = e2

    def __repr__(self):
        return f"Let({self.var_name}, {self.e1}, {self.e2})"

class If(Node):
    """
    An if form: (if e1 e2 e3)
    """
    def __init__(self, e1, e2, e3):
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    def __repr__(self):
        return f"If({self.e1}, {self.e2}, {self.e3})"

class For(Node):
    """
    A for loop: (for [loop_var range_n] body )
    We assume range_n is known at compile time.
    """
    def __init__(self, loop_var, range_n, body):
        self.loop_var = loop_var     # string
        self.range_n = range_n       # int or expression returning int
        self.body = body            # list of AST nodes

    def __repr__(self):
        return f"For({self.loop_var} in 0..{self.range_n}: {self.body})"

class Sample(Node):
    def __init__(self, var_name, distribution_expr):
        self.var_name = var_name
        self.distribution_expr = distribution_expr

    def __repr__(self):
        return f"Sample({self.var_name}, {self.distribution_expr})"

class Observe(Node):
    def __init__(self, distribution_expr, observed_value):
        self.distribution_expr = distribution_expr
        self.observed_value = observed_value

    def __repr__(self):
        return f"Observe({self.distribution_expr}, {self.observed_value})"

class Assign(Node):
    def __init__(self, var_name, expr):
        self.var_name = var_name
        self.expr = expr

    def __repr__(self):
        return f"Assign({self.var_name}, {self.expr})"