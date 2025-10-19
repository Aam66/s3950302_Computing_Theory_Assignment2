from itertools import permutations
import random
import time

def create_graph(n):
    graph = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance = random.randint(1, 100)
            graph[i][j] = distance
            graph[j][i] = distance
    return graph

def tsp(cost):
    num_nodes = len(cost)
    nodes = list(range(1, num_nodes))
    min_cost = float('inf')
    for perm in permutations(nodes):
        curr_cost = 0
        curr_node = 0
        for node in perm:
            curr_cost += cost[curr_node][node]
            curr_node = node
        curr_cost += cost[curr_node][0]
        min_cost = min(min_cost, curr_cost)
    return min_cost

if __name__ == "__main__":
    n = 11  # Adjust n, warn if >12
    if n > 12:
        print("Warning: n > 12 may result in very long computation time!")
    graph = create_graph(n)
    start_time = time.time()
    result = tsp(graph)
    end_time = time.time()
    print(f"Minimum Cost: {result}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")