# 2-approximation for metric TSP via MST preorder ("double-tree"):

# 1) Build an MST with Prim (dense O(n^2) version for simplicity).
# 2) Take a preorder traversal of the MST (each vertex once).
# 3) Compute the tour cost for that preorder (implicit shortcutting).


def _prim_mst_parent(cost):
    # Prim's MST (dense O(n^2)) → parent array (parent[0] = -1).
    n = len(cost)
    in_mst = [False] * n
    key = [float("inf")] * n
    parent = [-1] * n
    key[0] = 0

    # Repeatedly add the cheapest non-MST vertex
    for _ in range(n):
        u = -1
        best = float("inf")
        for v in range(n):
            if not in_mst[v] and key[v] < best:
                best = key[v]
                u = v
        if u == -1:            
            break
        in_mst[u] = True
        # Relax edges out of u
        for v in range(n):
            w = cost[u][v]
            if not in_mst[v] and 0 < w < key[v]:
                key[v] = w
                parent[v] = u
    return parent

def _mst_adj_from_parent(parent):
    # Undirected adjacency list from MST parent[] array.
    n = len(parent)
    adj = [[] for _ in range(n)]
    for v in range(1, n):
        u = parent[v]
        if u != -1:
            adj[u].append(v)
            adj[v].append(u)
    return adj

def _preorder(adj, start=0):
    # Iterative DFS preorder; each vertex appears exactly once.
    visited = [False] * len(adj)
    order = []
    stack = [start]
    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)
        # Push in reverse-sorted order so smaller indices pop first
        for v in sorted(adj[u], reverse=True):
            if not visited[v]:
                stack.append(v)
    return order

def _tour_cost(cost, order):
    # Cycle cost through 'order' and back to the start node.
    total = 0
    for i in range(len(order) - 1):
        total += cost[order[i]][order[i + 1]]
    total += cost[order[-1]][order[0]]
    return total

def approx_tsp(cost):
    # Metric TSP 2-approx via MST preorder (double-tree).
    parent = _prim_mst_parent(cost)
    adj = _mst_adj_from_parent(parent)
    order = _preorder(adj, start=0)
    return _tour_cost(cost, order)
