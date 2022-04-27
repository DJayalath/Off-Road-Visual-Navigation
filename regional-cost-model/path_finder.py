import numpy as np

def get_path(c_grid, NUM_STRIPS):

    # First row is source node, next NUM_STRIPS rows are bottom rank, next NUM_STRIPS rows are 2nd bottom rank, ...


    # first NUM_STRIPS are top rank, following NUM_STRIPS rows are 2nd top rank nodes, ... and so on down to bottom rank
    graph = np.array([[np.inf for _ in range(NUM_STRIPS ** 2)] for _ in range(NUM_STRIPS ** 2)])

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

            # Left and right movement is difficult to directly follow for a rover...
            
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
    from scipy.sparse.csgraph import reconstruct_path
    graph = csr_matrix(graph)

    # !!!! STart from middle of bottom row!
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=True, indices=NUM_STRIPS * (NUM_STRIPS - 1) + NUM_STRIPS // 2, return_predecessors=True)

    # Find top 3 paths (0 to)
    # print(dist_matrix)
    top_3_idx = dist_matrix[0 : NUM_STRIPS].argsort()[:3]

    # Points of interest
    # ALL of Middle row
    # LEFT below middle
    # RIGHT below middle
    # Pick top K trajectories from each?

    # Middle row
    paths, costs = [], []
    # mid = 6 * NUM_STRIPS (NUM_STRIPS ** 2) // 2
    mr = [39 - i for i in range(NUM_STRIPS)]
    for m in mr:
        p = [m]
        while p[-1] != -9999:
            p.append(predecessors[p[-1]])
        p = p[:-1]
        paths.append(p)
        costs.append(dist_matrix[m])
    
    # # LEFT below middle
    # lm = [mid - NUM_STRIPS * i for i in range(NUM_STRIPS // 2 - 2)]
    # for m in lm:
    #     p = [m]
    #     while p[-1] != -9999:
    #         p.append(predecessors[p[-1]])
    #     p = p[:-1]
    #     paths.append(p)
    #     costs.append(dist_matrix[m])

    # # RIGHT below middle
    # rm = [mid + NUM_STRIPS - 1 - NUM_STRIPS * i for i in range(NUM_STRIPS // 2 - 2)]
    # for m in rm:
    #     p = [m]
    #     while p[-1] != -9999:
    #         p.append(predecessors[p[-1]])
    #     p = p[:-1]
    #     paths.append(p)
    #     costs.append(dist_matrix[m])
        
    # p = [0]
    # while p[-1] != -9999:
    #     p.append(predecessors[p[-1]])

    #     # print()
    #     # p = p[:-2]
    #     # p = [(v - 1 - 15) for v in p] # Sub one AND 16 because we start from botto mrank middle
    # p = p[:-1]
    # print(p)

    print(paths)
    return np.array(paths), np.array(costs)