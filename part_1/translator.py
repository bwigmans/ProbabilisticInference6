import itertools
from ast_nodes import Constant, Variable, Let, If, For, Sample, Observe, Assign, Node


class GraphNode:
    def __init__(self, name, dist_info=None, parents=None, observed_value=None, deterministic_expr=None):
        """
        :param name: string name for this node
        :param dist_info: e.g. ('normal', mu, sigma) if sample
        :param observed_value: if not None, node is an observation
        :param deterministic_expr: if not None, node is a deterministic expression
        :param parents: list of other node names that this node depends on
        """
        self.name = name
        self.dist_info = dist_info
        self.observed_value = observed_value
        self.deterministic_expr = deterministic_expr
        self.parents = parents if parents else []

    def __repr__(self):
        return (f"GraphNode({self.name}, dist={self.dist_info}, obs={self.observed_value}, "
                f"parents={self.parents}, det_expr={self.deterministic_expr})")


class Translator:
    def __init__(self):
        self.counter = itertools.count()
        self.graph = {}
        self.env = {}  # env: AST var -> graph node name

    def translate_program(self, prog_ast):
        """
        Translate a top-level program (list of AST nodes) into a graph dict.
        """
        if isinstance(prog_ast, list):
            for node in prog_ast:
                self._translate_expr(node)
        else:
            self._translate_expr(prog_ast)
        return self.graph

    def _fresh_name(self, hint="tmp"):
        return f"{hint}_{next(self.counter)}"

    def _translate_expr(self, expr):

        if isinstance(expr, Constant):
            # Turn it into a node with a ('const', value) expression
            cname = self._fresh_name("const")
            self.graph[cname] = GraphNode(name=cname,
                                          deterministic_expr=('const', expr.value),
                                          parents=[])
            return cname

        elif isinstance(expr, Variable):
            if expr.name not in self.env:
                raise ValueError(f"Unbound variable {expr.name}")
            return self.env[expr.name]

        elif isinstance(expr, Let):
            # let v = e1 in e2
            node_e1 = self._translate_expr(expr.e1)
            self.env[expr.var_name] = node_e1
            node_e2 = self._translate_expr(expr.e2)
            return node_e2

        elif isinstance(expr, If):
            cond_name = self._translate_expr(expr.e1)
            then_name = self._translate_expr(expr.e2)
            else_name = self._translate_expr(expr.e3)
            merge = self._fresh_name("if")
            self.graph[merge] = GraphNode(
                name=merge,
                deterministic_expr=('if', cond_name, then_name, else_name),
                parents=[cond_name, then_name, else_name]
            )
            return merge

        elif isinstance(expr, For):
            # (for [loop_var range_n] body)
            # We assume range_n is known or a Constant. If it's not, you'd need more logic.
            if isinstance(expr.range_n, int):
                n = expr.range_n
            elif isinstance(expr.range_n, Constant):
                n = expr.range_n.value
            else:
                raise ValueError("Expecting a literal or Constant for range_n.")

            # We'll unroll
            last_value = None
            for i in range(n):
                # 1) bind loop_var to i
                i_node = self._fresh_name(expr.loop_var)
                self.graph[i_node] = GraphNode(
                    name=i_node,
                    deterministic_expr=('const', i),
                    parents=[]
                )
                # update env
                old_var_binding = self.env.get(expr.loop_var, None)
                self.env[expr.loop_var] = i_node

                # 2) translate the body
                # We might have multiple statements in expr.body
                for stmt in expr.body:
                    last_value = self._translate_expr(stmt)

                # 3) restore old var binding (if we want to scoping properly)
                if old_var_binding is not None:
                    self.env[expr.loop_var] = old_var_binding

            # If the loop's body yields no interesting final value, we might
            # just return last_value or a special placeholder.
            return last_value

        elif isinstance(expr, Sample):
            dist_node = self._translate_distribution_expr(expr.distribution_expr)
            var_name = expr.var_name
            self.graph[var_name] = GraphNode(
                name=var_name,
                dist_info=(dist_node,),
                parents=[dist_node]
            )
            self.env[var_name] = var_name
            return var_name

        elif isinstance(expr, Observe):
            dist_node = self._translate_distribution_expr(expr.distribution_expr)
            obs_name = self._fresh_name("obs")
            self.graph[obs_name] = GraphNode(
                name=obs_name,
                dist_info=(dist_node,),
                observed_value=expr.observed_value,
                parents=[dist_node]
            )
            return obs_name

        elif isinstance(expr, Assign):
            rhs_name = self._translate_expr(expr.expr)
            self.graph[expr.var_name] = GraphNode(
                name=expr.var_name,
                deterministic_expr=('id', rhs_name),
                parents=[rhs_name]
            )
            self.env[expr.var_name] = expr.var_name
            return expr.var_name

        if isinstance(expr, tuple):
            op = expr[0]

            # Let's define a set/list of recognized built-in ops:
            built_in_ops = {'+', '-', '*', '/', '>', '<', '==', '>=', '<='}

            if op in built_in_ops:
                # For a binary op, we expect something like (op, left_expr, right_expr)
                left_expr = expr[1]
                right_expr = expr[2]

                # Recursively translate the left and right subexpressions
                left_node = self._translate_expr(left_expr)
                right_node = self._translate_expr(right_expr)

                # Create a fresh name for this op node
                op_node_name = self._fresh_name(f"op_{op}")

                # Store a deterministic_expr describing the operation
                # e.g. ('>', left_node, right_node)
                self.graph[op_node_name] = GraphNode(
                    name=op_node_name,
                    deterministic_expr=(op, left_node, right_node),
                    parents=[left_node, right_node]
                )
                return op_node_name

        else:
            raise ValueError(f"Unknown expression type: {expr}")

    def _translate_distribution_expr(self, dist_expr):
        """
        dist_expr might be a tuple like ('normal', mu_expr, sigma_expr)
        or ('bernoulli', p_expr), etc.
        """
        if not isinstance(dist_expr, tuple):
            raise ValueError("Distribution expression must be a tuple.")
        dist_type = dist_expr[0]
        param_exprs = dist_expr[1:]

        param_nodes = []
        for p in param_exprs:
            if isinstance(p, Node):
                param_nodes.append(self._translate_expr(p))
            else:
                # treat as a constant
                param_nodes.append(self._translate_expr(Constant(p)))

        dist_node_name = self._fresh_name(dist_type)
        # store it as a deterministic node that collects the distribution info
        self.graph[dist_node_name] = GraphNode(
            name=dist_node_name,
            deterministic_expr=('dist', dist_type, param_nodes),
            parents=param_nodes
        )
        return dist_node_name
