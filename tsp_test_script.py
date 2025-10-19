import Brute_Force_TSP
import Approx_MST_TSP
import Nearest_Neighbor_TSP
import time

def run_tests():
    n_values = [5, 7, 9, 11]  # Adjust range as needed
    results = {"Brute-Force": [], "2-Approx MST": [], "Nearest-Neighbor": []}

    for n in n_values:
        print(f"\nTesting with n = {n}")
        
        # Brute-Force
        start_time = time.time()
        Brute_Force_TSP.tsp(Brute_Force_TSP.create_graph(n))
        end_time = time.time()
        results["Brute-Force"].append(end_time - start_time)
        if n > 12:
            print("Warning: Stopping Brute-Force due to high computation time.")

        # 2-Approx MST
        start_time = time.time()
        Approx_MST_TSP.approx_tsp(Approx_MST_TSP.create_graph(n))
        end_time = time.time()
        results["2-Approx MST"].append(end_time - start_time)

        # Nearest-Neighbor
        start_time = time.time()
        Nearest_Neighbor_TSP.find_min_route(Nearest_Neighbor_TSP.create_graph(n))
        end_time = time.time()
        results["Nearest-Neighbor"].append(end_time - start_time)

    # Print results table
    print("\nResults Table:")
    print("| n (Cities) | Brute-Force Time (s) | 2-Approx MST Time (s) | Nearest-Neighbor Time (s) |")
    print("|------------|----------------------|-----------------------|---------------------------|")
    for i, n in enumerate(n_values):
        print(f"| {n}          | {results['Brute-Force'][i]:.4f}               | {results['2-Approx MST'][i]:.4f}                | {results['Nearest-Neighbor'][i]:.4f}                    |")

    # Extrapolation for 30-minute limit (1800 seconds)
    print("\n30-Minute Limit Extrapolation:")
    # Simple linear extrapolation (adjust based on trend)
    if len(results["Brute-Force"]) > 1:
        brute_slope = (results["Brute-Force"][-1] - results["Brute-Force"][0]) / (n_values[-1] - n_values[0])
        brute_est_n = n_values[-1] + (1800 - results["Brute-Force"][-1]) / brute_slope if brute_slope > 0 else 12
        print(f"Brute-Force max n ~ {int(brute_est_n)} (estimated)")
    if len(results["2-Approx MST"]) > 1:
        approx_slope = (results["2-Approx MST"][-1] - results["2-Approx MST"][0]) / (n_values[-1] - n_values[0])
        approx_est_n = n_values[-1] + (1800 - results["2-Approx MST"][-1]) / approx_slope if approx_slope > 0 else 650
        print(f"2-Approx MST max n ~ {int(approx_est_n)} (estimated)")
    if len(results["Nearest-Neighbor"]) > 1:
        nn_slope = (results["Nearest-Neighbor"][-1] - results["Nearest-Neighbor"][0]) / (n_values[-1] - n_values[0])
        nn_est_n = n_values[-1] + (1800 - results["Nearest-Neighbor"][-1]) / nn_slope if nn_slope > 0 else 20000
        print(f"Nearest-Neighbor max n ~ {int(nn_est_n)} (estimated)")

if __name__ == "__main__":
    run_tests()