import numpy as np

# Example usage
# Define the graph
graph = {
    "z": ([], lambda _: np.random.binomial(1, 0.5)),  # z has no parents
    "y": (["z"], lambda values: np.random.normal(-1.0 if values["z"] == 0 else 1.0, 1.0))  # y depends on z
}

sigma_r = 1
D = 5
# Define the graph


graph2 = {
    "r": (
         [],
        lambda _: np.random.normal(0, sigma_r),
    ),
}

# Add z_i nodes
for i in range(1, D + 1):
    graph2[f"z_{i}"] = (
         ["r"],
        lambda parents, i=i: np.random.normal(0, np.exp(parents["r"])),
    )
