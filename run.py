"""
CSE 6140 - Project: Traveling Salesman Problem
--------------------------------------------------------------------------------
File Name: run.py
Author: Jitesh Jain/Christian DeRolf
Date: 11/30/29

Description:
    This script is the main executable for the TSP Project. It implements three
    algorithms to solve the Traveling Salesman Problem:
    
    1. Brute Force (BF): 
       - Exhaustively searches all permutations.
       - Returns the optimal solution if it finishes within the cutoff time.
       - If time runs out, returns the best solution found so far.
       
    2. Approximation (Approx):
       - Uses a Minimum Spanning Tree (MST) based approach.
       - Guaranteed to be within a factor of 2 of the optimal solution.
       - Runs efficiently (polynomial time).
       
    3. Local Search (LS):
       - Implements Simulated Annealing with 2-Opt swaps.

Usage:
    python run.py -inst <filename> -alg [BF | Approx | LS] -time <cutoff> [-seed <seed>]
    
    Example: 
    python run.py -inst DATA/Atlanta.tsp -alg LS -time 60 -seed 42
"""

import sys
import time
import math
import os
import random
import itertools
import numpy as np

# ==========================================
# 1. Data Parsing & Preprocessing
# ==========================================

def parse_tsp_file(filename):
    """
    Reads the .tsp file.
    Returns:
        ids (list): Node IDs.
        coords (np.array): (N, 2) array of coordinates.
    """
    print(f"[INFO] Loading dataset: {filename}")
    ids = []
    coords = []
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[ERROR] File {filename} not found.")
        sys.exit(1)
    
    node_section = False
    for line in lines:
        line = line.strip()
        if line == "NODE_COORD_SECTION":
            node_section = True
            continue
        if line == "EOF":
            break
        if node_section:
            parts = line.split()
            if len(parts) >= 3:
                ids.append(int(parts[0]))
                coords.append([float(parts[1]), float(parts[2])])
    
    return ids, np.array(coords)

def compute_distance_matrix(coords):
    """
    Computes NxN distance matrix using Euclidean distance rounded to nearest int.
    """
    # Vectorized calculation using Numpy broadcasting
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(delta**2, axis=-1))
    return np.round(dists).astype(int)

# ==========================================
# 2. Algorithm: Brute Force (BF)
# ==========================================

def algo_brute_force(ids, dist_matrix, cutoff_time):
    """
    Tries every permutation. Stops if cutoff_time is reached.
    """
    print(f"[INFO] Running Brute Force (Cutoff: {cutoff_time}s)...")
    start_time = time.time()
    n = len(ids)
    
    # We fix the first node (index 0) and permute the rest to reduce complexity
    nodes_to_visit = list(range(1, n))
    
    best_cost = float('inf')
    best_tour_indices = list(range(n))
    status = "COMPLETED" 

    # itertools.permutations returns a generator (memory efficient)
    for p in itertools.permutations(nodes_to_visit):
        
        # Check time limit
        if (time.time() - start_time) > cutoff_time:
            print("[INFO] Time limit reached. Stopping Brute Force.")
            status = "CUTOFF"
            break
            
        current_tour_indices = [0] + list(p)
        
        # Calculate cost
        current_cost = 0
        valid = True
        for i in range(n - 1):
            current_cost += dist_matrix[current_tour_indices[i]][current_tour_indices[i+1]]
            # Pruning: If we already exceed best_cost, stop checking this path
            if current_cost >= best_cost: 
                valid = False
                break
        
        if valid:
            # Add return to start
            current_cost += dist_matrix[current_tour_indices[-1]][0]
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour_indices = list(current_tour_indices)

    final_tour_ids = [ids[i] for i in best_tour_indices]
    return final_tour_ids, best_cost, status

# ==========================================
# 3. Algorithm: Approximation (MST)
# ==========================================

def algo_approx_mst(ids, dist_matrix):
    """
    2-Approximation using Prim's MST + Preorder Walk.
    """
    print("[INFO] Running MST Approximation...")
    n = len(ids)
    if n == 0: return [], 0, "COMPLETED"

    # Prim's Algorithm
    min_vals = np.full(n, np.inf)
    min_vals[0] = 0
    parent = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    
    for _ in range(n):
        # Find closest unvisited node
        current_min_vals = np.where(visited, np.inf, min_vals)
        u = np.argmin(current_min_vals)
        visited[u] = True
        
        # Update neighbors
        mask = (~visited) & (dist_matrix[u] < min_vals)
        min_vals[mask] = dist_matrix[u][mask]
        parent[mask] = u

    # Build Adjacency List
    mst_adj = [[] for _ in range(n)]
    for i in range(1, n):
        mst_adj[parent[i]].append(i)
        mst_adj[i].append(parent[i])

    # DFS Preorder Walk
    tour_indices = []
    visited_dfs = np.zeros(n, dtype=bool)
    stack = [0]
    while stack:
        u = stack.pop()
        if not visited_dfs[u]:
            visited_dfs[u] = True
            tour_indices.append(u)
            for v in sorted(mst_adj[u], reverse=True):
                if not visited_dfs[v]:
                    stack.append(v)
    
    # Calculate Cost
    tour_arr = np.array(tour_indices)
    cost = np.sum(dist_matrix[tour_arr, np.roll(tour_arr, -1)])
    final_tour_ids = [ids[i] for i in tour_indices]
    
    return final_tour_ids, cost, "COMPLETED"

# ==========================================
# 4. Algorithm: Local Search (SA)
# ==========================================

def algo_local_search(ids, dist_matrix, cutoff_time, seed):
    """
    Simulated Annealing with 2-Opt swaps.
    """
    print(f"[INFO] Running Local Search (Seed: {seed}, Cutoff: {cutoff_time}s)...")
    random.seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    n = len(ids)
    
    # Helper for cost calculation
    def get_cost(t): return np.sum(dist_matrix[t, np.roll(t, -1)])
    
    # Initialize Random Solution
    tour = np.arange(n)
    np.random.shuffle(tour)
    
    current_cost = get_cost(tour)
    best_tour = tour.copy()
    best_cost = current_cost
    
    # SA Parameters
    T = 1000.0
    alpha = 0.9995
    min_T = 1e-4
    iter_count = 0
    
    # Simulated Annealing Loop
    while T > min_T:
        # Check time limit
        if (time.time() - start_time) >= cutoff_time:
            break
            
        iter_count += 1
        
        # 2-Opt Selection
        a = random.randint(0, n-1)
        b = random.randint(0, n-1)
        if a == b: continue
        i, j = (a, b) if a < b else (b, a)
        if i == 0 and j == n-1: continue
            
        # Delta Calculation
        n_i_p, n_i, n_j, n_j_n = tour[i-1], tour[i], tour[j], tour[(j+1)%n]
        delta = (dist_matrix[n_i_p, n_j] + dist_matrix[n_i, n_j_n]) - \
                (dist_matrix[n_i_p, n_i] + dist_matrix[n_j, n_j_n])
        
        # Metropolis Acceptance
        if delta < 0 or random.random() < math.exp(-delta / T):
            tour[i:j+1] = tour[i:j+1][::-1]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = tour.copy()
        
        if iter_count % 100 == 0: 
            T *= alpha
            
    print(f"[INFO] Local Search finished.")
    final_tour_ids = [ids[x] for x in best_tour]
    return final_tour_ids, best_cost, "COMPLETED"

# ==========================================
# 5. Main Execution & Output
# ==========================================

def write_output(filename, method, cutoff, seed, cost, tour):
    """
    Saves the solution to the 'output' folder.
    """
    # Create output directory per deliverables
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base = os.path.basename(filename).split('.')[0]
    
    # Naming convention based on PDF
    if method == "BF": 
        out_name = f"{base}_BF_{cutoff}.sol"
    elif method == "Approx": 
        # Using seed in filename for Approx to keep format consistent
        out_name = f"{base}_Approx_{seed}.sol"
    elif method == "LS": 
        out_name = f"{base}_LS_{cutoff}_{seed}.sol"
    else:
        out_name = "output.sol"
    
    full_path = os.path.join(output_dir, out_name)
    
    with open(full_path, 'w') as f:
        f.write(f"{cost}\n")
        f.write(",".join(map(str, tour)))
        
    print(f"[INFO] Solution saved to: {full_path}")
    return full_path

def main():
    # Argument Parsing
    args = sys.argv
    filename, method, cutoff, seed = None, None, 0, 0
    
    for i in range(len(args)):
        if args[i] == "-inst": filename = args[i+1]
        elif args[i] == "-alg": method = args[i+1]
        elif args[i] == "-time": cutoff = int(args[i+1])
        elif args[i] == "-seed": seed = int(args[i+1])

    if not filename or not method:
        print("Usage: python run.py -inst <file> -alg [BF|Approx|LS] -time <seconds> [-seed <seed>]")
        return

    # Load Data
    ids, coords = parse_tsp_file(filename)
    dist_matrix = compute_distance_matrix(coords)
    
    start = time.time()
    status = ""
    best_tour = []
    best_cost = float('inf')
    
    # Execute Algorithm
    if method == "Approx":
        best_tour, best_cost, status = algo_approx_mst(ids, dist_matrix)
    elif method == "LS":
        best_tour, best_cost, status = algo_local_search(ids, dist_matrix, cutoff, seed)
    elif method == "BF":
        best_tour, best_cost, status = algo_brute_force(ids, dist_matrix, cutoff)
    else:
        print(f"[ERROR] Unknown method: {method}")
        return
    
    dur = time.time() - start
    
    # Write File
    write_output(filename, method, cutoff, seed, best_cost, best_tour)
    
    # Print formatted result for Bash script parsing
    print(f"[RESULT] time={dur:.4f} cost={int(best_cost)} status={status}")

if __name__ == "__main__":
    main()
