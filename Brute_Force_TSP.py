
# Exact solver for TSP using brute-force permutations.

# - Starts at node 0, tries every permutation of the remaining nodes,
#   and returns the minimum tour cost.
# - Intended only for very small n (factorial time).


from itertools import permutations

def tsp_min_cost(cost):
    # Exact TSP via brute-force permutations from start node 0. O(n!).
    n = len(cost)
    # Permute all nodes except the fixed start (0)
    nodes = list(range(1, n))
    best = float("inf")

    for perm in permutations(nodes):
        total = 0
        curr = 0
        # Walk 0 -> perm[0] -> ... -> perm[-1]
        for v in perm:
            total += cost[curr][v]
            curr = v
        # Close the cycle back to 0
        total += cost[curr][0]
        if total < best:
            best = total

    return best
