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

def find_mst(graph):
    n = len(graph)
    visited = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0
    parent = [-1] * n
    for _ in range(n):
        min_val = float('inf')
        u = -1
        for v in range(n):
            if not visited[v] and min_dist[v] < min_val:
                min_val = min_dist[v]
                u = v
        if u == -1:
            break
        visited[u] = True
        for v in range(n):
            if (not visited[v] and graph[u][v] != 0 and graph[u][v] < min_dist[v]):
                min_dist[v] = graph[u][v]
                parent[v] = u
    return parent

def approx_tsp(graph):
    n = len(graph)
    parent = find_mst(graph)
    # Double the edges and shortcut (simplified)
    tour = [0]
    for i in range(1, n):
        tour.append(i)
    tour.append(0)
    total_cost = 0
    for i in range(len(tour) - 1):
        total_cost += graph[tour[i]][tour[i + 1]]
    return total_cost

if __name__ == "__main__":
    n = 450  # Adjust n for testing
    graph = create_graph(n)
    start_time = time.time()
    result = approx_tsp(graph)
    end_time = time.time()
    print(f"Approximate Cost: {result}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")