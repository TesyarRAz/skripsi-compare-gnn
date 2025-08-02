import numpy as np
import heapq
import model

def generate_sample(num_nodes: int, edge_values_raw):
    edges = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)

    edge_values = np.zeros((num_nodes, num_nodes))

    for v in edge_values_raw:
        edge_values[v.x, v.y] = v.distance

    nodes = np.arange(num_nodes)

    return model.DotDict(
        nodes=nodes,
        edges=[edges],
        edges_values=[edge_values],
    )


def generate_sample_from_coords(coords):
    num_nodes = len(coords)

    # Fully connected graph (no self-loop)
    edges = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    # edges = edges.array()

    # Distance matrix
    edge_values = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_values[i, j] = haversine_distance(coords[i], coords[j])

    # Node indices
    nodes = np.arange(num_nodes)

    return model.DotDict(
        nodes=nodes,
        nodes_coord=coords,
        edges=[edges],
        edges_values=[edge_values],
    )

def haversine_distance(coord1, coord2):
    R = 6371  # Earth radius in km
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def simple_beamsearch(prob_matrix, edge_dist, beam_size=3, start_node=0):
    """
    Beam Search sederhana untuk TSP.

    Args:
        prob_matrix (Tensor): Probabilitas antar-node, shape: (V, V)
        edge_dist (Tensor): Jarak antar-node, shape: (V, V)
        beam_size (int): Ukuran beam
        start_node (int): Node awal

    Returns:
        best_path (list): Rute TSP terbaik (urutan node)
        best_score (float): Total skor (negatif untuk minimisasi)
    """
    V = prob_matrix.shape[0]

    # Beam awal: [(score, path, visited)]
    beam = [(-0.0, [start_node], set([start_node]))]

    for step in range(V - 1):
        new_beam = []
        for score, path, visited in beam:
            last = path[-1]
            for next_node in range(V):
                if next_node in visited:
                    continue
                new_path = path + [next_node]
                new_visited = visited | {next_node}
                # Kombinasi: pakai -prob dan +jarak sebagai penalti (karena kita min skor)
                prob_score = prob_matrix[last, next_node].item()
                dist_score = edge_dist[last, next_node].item()
                new_score = score - prob_score + dist_score * 0.01  # tweak weight
                new_beam.append((new_score, new_path, new_visited))

        # Ambil beam terbaik
        beam = heapq.nsmallest(beam_size, new_beam, key=lambda x: x[0])

    # Tambahkan edge balik ke start
    final_paths = []
    for score, path, visited in beam:
        back_cost = edge_dist[path[-1], path[0]].item()
        final_score = score + back_cost * 0.01
        final_paths.append((final_score, path))

    # Ambil path terbaik
    best_score, best_path = min(final_paths, key=lambda x: x[0])
    return best_path, best_score

# Konversi list of node → adjacency matrix (mask)
def tour_nodes_to_adj_matrix(tour):
    V = len(tour)
    adj = np.zeros((V, V))
    for i in range(V):
        u, v = tour[i], tour[(i+1)%V]
        adj[u][v] = 1
    return adj

def path_cost(path, dist):
    return sum(dist[path[i]][path[i+1]] for i in range(len(path)-1)) + dist[path[-1]][path[0]]