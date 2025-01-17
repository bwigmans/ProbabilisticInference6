import math, random
from collections import deque
import numpy as np
from ast_nodes import Constant
class ImportanceSampler:
    def __init__(self, graph):
        self.graph = graph
        self.sorted_nodes = self.topological_sort()

    def topological_sort(self):
        in_degree = { v:0 for v in self.graph }
        for v, node in self.graph.items():
            for p in node.parents:
                in_degree[v] += 1
        queue = deque([n for n in in_degree if in_degree[n] == 0])
        result = []
        while queue:
            v = queue.popleft()
            result.append(v)
            for w, node_w in self.graph.items():
                if v in node_w.parents:
                    in_degree[w] -= 1
                    if in_degree[w] == 0:
                        queue.append(w)
        return result

    def sample_one(self):
        env = {}
        log_weight = 0.0
        for var_name in self.sorted_nodes:
            node = self.graph[var_name]

            # Deterministic node?
            if node.deterministic_expr is not None:
                val = self.eval_deterministic_expr(node.deterministic_expr, env)
                env[var_name] = val

            # Sample / Observe node?
            elif node.dist_info is not None:
                dist_node_name = node.dist_info[0]
                # Evaluate distribution
                dist_val = self.eval_deterministic_expr(self.graph[dist_node_name].deterministic_expr, env)
                # dist_val is like ('dist', 'normal', [ param_node_names ... ])
                dist_type = dist_val[1]
                param_list = dist_val[2]
                # param_list are the node names for the distribution parameters:
                param_values = [env[x] for x in param_list]

                if node.observed_value is None:
                    # sample
                    sample_val, lp = self.sample_from_distribution(dist_type, param_values)
                    env[var_name] = sample_val
                    # If you're doing prior-sampling, you might not add to log_weight.
                    # If you're doing proper IS, you'd incorporate the proposal ratio.
                else:
                    # observe
                    obs_val = node.observed_value
                    lp = self.log_prob_of_distribution(dist_type, param_values, obs_val)
                    env[var_name] = obs_val
                    log_weight += lp

        return env, log_weight

    def eval_deterministic_expr(self, expr, env):
        op = expr[0]
        if op == 'const':
            return expr[1]
        elif op == 'if':
            cond_name, then_name, else_name = expr[1], expr[2], expr[3]
            cond_val = env[cond_name]
            return env[then_name] if cond_val else env[else_name]
        elif op == 'dist':
            # e.g. ('dist', 'normal', ['nodeA','nodeB'])
            return expr
        elif op == 'id':
            return env[expr[1]]
        elif op in ['+', '-', '*', '>', '>=', '<', '<=', '==']:
            left_node = expr[1]
            right_node = expr[2]
            left_val = env[left_node]
            right_val = env[right_node]
            if op == '+':
                return left_val + right_val
            elif op == '-':
                return left_val - right_val
            elif op == '*':
                return left_val * right_val
            elif op == '>':
                return left_val > right_val
            elif op == '>=':
                return left_val >= right_val
            elif op == '<':
                return left_val < right_val
            elif op == '<=':
                return left_val <= right_val
            elif op == '==':
                return left_val == right_val
        else:
            raise ValueError(f"Unknown deterministic op: {op}")

    def sample_from_distribution(self, dist_type, params):
        if dist_type == 'normal':
            mu, sigma = params
            val = random.gauss(mu, sigma)
            lp = self.log_prob_normal(val, mu, sigma)
            return val, lp
        elif dist_type == 'bernoulli':
            p = params[0]
            val = 1 if random.random()<p else 0
            lp = math.log(p if val==1 else (1-p))
            return val, lp
        elif dist_type == 'poisson':
            lam = params[0]
            val = np.random.poisson(lam)
            lp = -lam + val * math.log(lam) - math.lgamma(val + 1)
            return val, lp
        elif dist_type == 'beta':
            alpha, beta_val = params
            val = random.betavariate(alpha, beta_val)
            lp = (math.lgamma(alpha + beta_val)
                  - math.lgamma(alpha)
                  - math.lgamma(beta_val)
                  + (alpha - 1) * math.log(val)
                  + (beta_val - 1) * math.log(1 - val))
            return val, lp
        
        elif dist_type =='half-cauchy':
            gamma = params[0]
            # Sample from the Cauchy distribution and take the absolute value to ensure positivity
            val = np.random.standard_cauchy() * gamma
            val = abs(val)
            
            # Log probability for the Half-Cauchy distribution
            lp = math.log(2 / (math.pi * gamma * (1 + (val / gamma) ** 2)))
            return val, lp

        elif dist_type =='lognormal':
            mu, sigma = params
            val = np.random.lognormal(mu, sigma)
            lp = self.log_prob_lognormal(val, mu, sigma)
            return val, lp
            
        else:
            raise NotImplementedError(f"Distribution {dist_type} not supported.")

    def log_prob_of_distribution(self, dist_type, params, x):
        if dist_type == 'normal':
            mu, sigma = params
            return self.log_prob_normal(x, mu, sigma)
        elif dist_type == 'bernoulli':
            p = params[0]
            return math.log(p if x==1 else (1-p))
        elif dist_type == 'poisson':
            lam = params[0]
            if isinstance(x, Constant):
                x = x.value
            return -lam + x * math.log(lam) - math.lgamma(x + 1)
        elif dist_type == 'beta':
            alpha, beta_val = params
            if x <= 0 or x >= 1:
                return float('-inf')
            return (math.lgamma(alpha + beta_val)
                    - math.lgamma(alpha)
                    - math.lgamma(beta_val)
                    + (alpha - 1) * math.log(x)
                    + (beta_val - 1) * math.log(1 - x))
        elif dist_type == 'lognormal':
            mu, sigma = params
            return self.log_prob_lognormal(x, mu, sigma)
        else:
            raise NotImplementedError(f"Distribution {dist_type} not supported.")

    def log_prob_normal(self, x, mu, sigma):
        if isinstance(x, Constant):
            x = x.value
        if isinstance(mu, Constant):
            mu = mu.value
        if isinstance(sigma, Constant):
            sigma = sigma.value
        return -0.5*math.log(2*math.pi*sigma*sigma) - 0.5*((x - mu)/sigma)**2

    def log_prob_lognormal(self, x, mu, sigma):
        if isinstance(x, Constant):
            x = x.value
        if isinstance(mu, Constant):
            mu = mu.value
        if isinstance(sigma, Constant):
            sigma = sigma.value
        return -0.5*math.log(2*math.pi) -math.log(x*sigma) - 0.5*((math.log(x) - mu)/sigma)**2