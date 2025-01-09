import numpy as np
from collections import defaultdict, deque
from graphs import graph2 as graph

class ProbabilisticGraphSampler:
    def __init__(self, graph):
        """
        Initialize the sampler with a graph.
        :param graph: 
            keys : random variables
            values : (parents, expression).
                      Example:
                      {
                          "z": ([], lambda: np.random.binomial(1, 0.5)),
                          "y": (["z"], lambda values: np.random.normal(-1.0 if values["z"] == 0 else 1.0, 1.0))
                      }
        """
        self.graph = graph
        self.values = {}
        self.sorted_nodes = self.topological_sort()

    def topological_sort(self):
        """
        Perform a topological sort of the graph.
        :return: A list of nodes in topological order.
        """
        # Build the dependency graph
        in_degree = defaultdict(int)  # Count of incoming edges for each node
        adj_list = defaultdict(list)  # Adjacency list for graph traversal

        for node, (parents, _) in self.graph.items():
            for parent in parents:
                in_degree[node] += 1
                adj_list[parent].append(node)

        # Collect nodes with no incoming edges
        queue = deque([node for node in self.graph if in_degree[node] == 0])
        sorted_nodes = []

        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)

            # Reduce in-degree for child nodes
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_nodes) != len(self.graph):
            print(sorted_nodes)
            raise ValueError("Graph contains a cycle!")

        return sorted_nodes

    def sample_trace(self):
        """
        Sample a single trace by evaluating all nodes in topological order.
        :return: A dictionary of sampled values for all nodes.
        """
        self.values.clear()  # Clear previous sampled values
        for node in self.sorted_nodes:
            parents, expression = self.graph[node]
            # Gather parent values
            parent_values = {p: self.values[p] for p in parents}
            # Evaluate the current node
            self.values[node] = expression(parent_values)
        return self.values

def main():
    # Create the sampler
    sampler = ProbabilisticGraphSampler(graph)

    # Sample traces
    print("Sampling traces:")
    for _ in range(5):
        trace = sampler.sample_trace()
        print(trace)

if __name__ == "__main__":
    main()
