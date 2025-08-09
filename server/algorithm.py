import numpy as np
import utils
import random
from itertools import combinations

def ant_colony(cities, num_ants=20, num_iterations=100, alpha=1.0, beta=5.0, rho=0.5, Q=100):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = utils.haversine_distance(cities[i], cities[j])

    # Initialize pheromone levels
    pheromone = np.ones((n, n))
    best_path = None
    best_cost = float('inf')

    def path_cost(path):
        return sum(dist[path[i]][path[i+1]] for i in range(n-1)) + dist[path[-1]][path[0]]

    for iteration in range(num_iterations):
        all_paths = []
        all_costs = []

        for ant in range(num_ants):
            path = [0]
            visited = set(path)

            for _ in range(n - 1):
                current = path[-1]

                # Probabilistic selection
                probabilities = []
                for j in range(n):
                    if j not in visited:
                        tau = pheromone[current][j] ** alpha
                        eta = (1 / dist[current][j]) ** beta if dist[current][j] > 0 else 1e6
                        probabilities.append((j, tau * eta))

                if not probabilities:
                    break

                total = sum(p[1] for p in probabilities)
                probs = [(j, p / total) for j, p in probabilities]
                r = random.random()
                cumulative = 0
                for j, prob in probs:
                    cumulative += prob
                    if r <= cumulative:
                        path.append(j)
                        visited.add(j)
                        break

            cost = path_cost(path)
            all_paths.append(path)
            all_costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_path = path.copy()

        # Evaporate
        pheromone *= (1 - rho)

        # Update pheromone
        for path, cost in zip(all_paths, all_costs):
            for i in range(n):
                a, b = path[i], path[(i + 1) % n]
                pheromone[a][b] += Q / cost
                pheromone[b][a] += Q / cost  # symmetric

    return best_path, best_cost

def held_karp(cities):
    """
    Implementasi algoritma Held-Karp untuk menyelesaikan TSP menggunakan dynamic programming.
    Time complexity: O(n^2 * 2^n)
    Space complexity: O(n * 2^n)
    
    Args:
        cities: List koordinat kota [(lat1, lon1), (lat2, lon2), ...]
    
    Returns:
        tuple: (best_path, best_cost)
    """
    n = len(cities)
    
    # Buat distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = utils.haversine_distance(cities[i], cities[j])
    
    # dp[mask][i] = minimum cost untuk mengunjungi semua kota dalam mask dan berakhir di kota i
    # mask adalah bitmask yang merepresentasikan subset kota yang sudah dikunjungi
    dp = {}
    parent = {}
    
    # Base case: mulai dari kota 0, hanya mengunjungi kota 0
    dp[(1, 0)] = 0  # mask=1 (binary: 0001) berarti hanya kota 0 yang dikunjungi
    
    # Iterasi untuk setiap subset size dari 2 hingga n
    for subset_size in range(2, n + 1):
        # Generate semua subset dengan ukuran subset_size yang mengandung kota 0
        for subset in combinations(range(1, n), subset_size - 1):
            # Tambahkan kota 0 ke subset
            subset = (0,) + subset
            mask = sum(1 << i for i in subset)
            
            # Untuk setiap kota terakhir dalam subset
            for last in subset:
                if last == 0:  # Skip kota awal
                    continue
                
                min_cost = float('inf')
                best_prev = -1
                
                # Coba semua kemungkinan kota sebelumnya
                prev_mask = mask ^ (1 << last)  # Remove last city from mask
                for prev in subset:
                    if prev != last and (prev_mask, prev) in dp:
                        cost = dp[(prev_mask, prev)] + dist[prev][last]
                        if cost < min_cost:
                            min_cost = cost
                            best_prev = prev
                
                if best_prev != -1:
                    dp[(mask, last)] = min_cost
                    parent[(mask, last)] = best_prev
    
    # Temukan solusi optimal (kembali ke kota 0)
    final_mask = (1 << n) - 1  # Semua kota sudah dikunjungi
    min_cost = float('inf')
    best_last = -1
    
    for last in range(1, n):
        if (final_mask, last) in dp:
            cost = dp[(final_mask, last)] + dist[last][0]
            if cost < min_cost:
                min_cost = cost
                best_last = last
    
    # Rekonstruksi path
    if best_last == -1:
        return None, float('inf')
    
    path = []
    current_mask = final_mask
    current = best_last
    
    # Trace back dari akhir ke awal
    while (current_mask, current) in parent:
        path.append(current)
        next_current = parent[(current_mask, current)]
        current_mask ^= (1 << current)  # Remove current city from mask
        current = next_current
    
    path.append(0)  # Tambahkan kota awal
    path.reverse()  # Reverse untuk mendapatkan urutan yang benar
    
    return path, min_cost