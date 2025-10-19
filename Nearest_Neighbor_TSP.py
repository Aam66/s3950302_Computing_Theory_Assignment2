from collections import defaultdict
import random
import time

INT_MAX = 2147483647

def create_graph(n):
    graph = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance = random.randint(1, 100)
            graph[i][j] = distance
            graph[j][i] = distance
    return graph

def find_min_route(tsp):
    sum_cost = 0
    counter = 0
    j = 0
    i = 0
    min_dist = INT_MAX
    visited = defaultdict(int)
    visited[0] = 1
    route = [0] * len(tsp)
    while i < len(tsp) and j < len(tsp[i]):
        if counter >= len(tsp[i]) - 1:
            break
        if j != i and visited[j] == 0:
            if tsp[i][j] < min_dist:
                min_dist = tsp[i][j]
                route[counter] = j + 1
        j += 1
        if j == len(tsp[i]):
            sum_cost += min_dist
            min_dist = INT_MAX
            visited[route[counter] - 1] = 1
            j = 0
            i = route[counter] - 1
            counter += 1
    i = route[counter - 1] - 1
    for j in range(len(tsp)):
        if i != j and tsp[i][j] < min_dist:
            min_dist = tsp[i][j]
            route[counter] = j + 1
    sum_cost += min_dist
    print("Minimum Cost is:", sum_cost)
    return sum_cost

if __name__ == "__main__":
    n = 9000  # Adjust n for testing
    graph = create_graph(n)
    start_time = time.time()
    find_min_route(graph)
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.4f} seconds")