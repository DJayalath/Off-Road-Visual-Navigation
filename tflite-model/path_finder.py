################################################################################
#
# (c) Copyright University of Southampton, 2022
#
# Copyright in this software belongs to the University of Southampton,
# Highfield, University Road, Southampton, SO17 1BJ, United Kingdom
#
# Created By : Dulhan Jayalath
# Date : 2022/04/27
# Project : Real-time Neural Visual Navigation for Autonomous Off-Road Robots
#
################################################################################

# ------------------------------------------------------------------------------
# Calculates all trajectories from robot position to target nodes at horizon.
#-------------------------------------------------------------------------------

import numpy as np

def get_path(c_grid, NUM_STRIPS):

    # First NUM_STRIPS are top rank, following NUM_STRIPS rows are 2nd top rank nodes, ... and so on down to bottom rank
    graph = [[np.inf for _ in range(NUM_STRIPS ** 2 + 1)] for _ in range(NUM_STRIPS ** 2 + 1)]
    graph = np.array(graph)

    # Let the start nodes be the central two nodes on the bottom rank (for even N graphs)
    # Add connection from start node to nodes adjacent and above
    # FIXME: Designed for 8x8 grids ONLY
    graph[-1, NUM_STRIPS * (NUM_STRIPS - 1) + 3] = c_grid[NUM_STRIPS - 1, 3]
    graph[-1, NUM_STRIPS * (NUM_STRIPS - 1) + 4] = c_grid[NUM_STRIPS - 1, 4]

    # Define other edges
    for y in range(NUM_STRIPS):
        for x in range(NUM_STRIPS):

            idx = y * NUM_STRIPS + x

            # Above
            if y - 1 >= 0:
                pidx = (y - 1) * NUM_STRIPS + x
                graph[idx, pidx] = c_grid[y - 1, x]

                # Above left
                if x - 1 >= 0:
                    pidx = (y - 1) * NUM_STRIPS + x - 1
                    graph[idx, pidx] = c_grid[y - 1, x - 1]

                # Above right
                if x + 1 < NUM_STRIPS:
                    pidx = (y - 1) * NUM_STRIPS + x + 1
                    graph[idx, pidx] = c_grid[y - 1, x + 1]

            # Assume rover can zero-turn so allow left and right movements
            
            # Left
            if x - 1 >= 0:
                pidx = y * NUM_STRIPS + x - 1
                graph[idx, pidx] = c_grid[y, x - 1]

            # Right
            if x + 1 < NUM_STRIPS:
                pidx = y * NUM_STRIPS + x + 1
                graph[idx, pidx] = c_grid[y, x + 1]

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    graph = csr_matrix(graph)

    # Get shortest paths starting from central two nodes in bottom rank
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=True, indices=[60, 59], return_predecessors=True)

    paths, costs = [], []

    # For each starting position
    for k in range(2):
        preds = predecessors[k]
        dists = dist_matrix[k]

        # List of all nodes in target rank
        mr = [39 - i for i in range(NUM_STRIPS)]

        # Find paths to all target nodes
        for m in mr:
            p = [m]
            while p[-1] != -9999:
                p.append(preds[p[-1]])
            p = p[:-1]
            paths.append(p)
            costs.append(dists[m])
    
    return np.array(paths), np.array(costs)