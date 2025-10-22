# Nearest-neighbour heuristic for TSP:

# - Start at node 0.
# - Repeatedly go to the nearest unvisited node.
# - Return to start and report the tour cost.
# - Simple O(n^2) baseline for dense graphs.


def nearest_neighbor_cost(cost, start=0):
    # Nearest-neighbour heuristic (dense O(n^2)).
    n = len(cost)
    visited = [False] * n
    order = [start]
    visited[start] = True
    curr = start

    for _ in range(n - 1):
        nxt = None
        best = float("inf")
        # Scan all unvisited nodes; pick the closest
        for v in range(n):
            if not visited[v] and 0 < cost[curr][v] < best:
                best = cost[curr][v]
                nxt = v
        order.append(nxt)
        visited[nxt] = True
        curr = nxt

    # Sum edges along the path + return-to-start edge
    total = 0
    for i in range(n - 1):
        total += cost[order[i]][order[i + 1]]
    total += cost[order[-1]][order[0]]
    return total
