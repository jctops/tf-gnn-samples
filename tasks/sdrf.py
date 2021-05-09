import networkx as nx
import numpy as np

softmax = lambda x: np.exp(x)/sum(np.exp(x))

def uma_forman_parallel(A, N, return_extra=True):
    NC2 = int(N*(N-1)/2)+1
    E = np.zeros((NC2,N,N))
    no_of_edges = np.zeros(NC2)
    no_of_edges[0] = N
    i = 1
    mapping = {}
    for x in range(N):
        for y in [n for n in range(N) if n > x]:
            if A[x,y]:
                E[i,x,y] = E[i,y,x] = -1
                no_of_edges[i] = N-1
            else:
                E[i,x,y] = E[i,y,x] = 1
                no_of_edges[i] = N+1
            mapping[(x,y)] = i
            i += 1
    AE = A+E
    
    AE2 = np.matmul(AE,AE)
    degrees = np.diagonal(AE2, axis1=1, axis2=2)
    degrees_reshaped = degrees.reshape((NC2,-1,N))
    degrees_reshaped_T = degrees.reshape((NC2,N,-1))
    squares = np.matmul(AE2,AE) - degrees_reshaped - degrees_reshaped_T + 1
    
    if return_extra:
        return (4 - degrees_reshaped - degrees_reshaped_T + 3*AE2 + squares)*AE, AE, no_of_edges, mapping
    return (4 - degrees_reshaped - degrees_reshaped_T + AE2 + squares)*A


def focused_stochastic_ricci_step(G, A, N, curvature_fn, target_curvature=0, give_graphs_ricci_attribute=True,
                          prioritise_betweenness=False, consider_positivity=True):
    curvatures, AE, no_of_edges, mapping = curvature_fn(A, N)
    
    candidates = {}
    x = AE*target_curvature - curvatures
    if not consider_positivity:
        x = (abs(x) + x) / 2
    mses = (x**2).sum(axis=2).sum(axis=1)/no_of_edges
    mses = mses[0] - mses
    
    C = curvatures[0]
    minima = np.argwhere(C == np.min(C))
    ijs = [(mini[0],mini[1]) for mini in minima]
    
    if prioritise_betweenness:
        betweenness = {k:v for (k,v) in nx.algorithms.centrality.edge_betweenness_centrality(G).items() if k in ijs}
        betweenness = {k: v for k, v in sorted(betweenness.items(), key=lambda item: -item[1])}
        ijs = list(betweenness.keys())
    else:
        ijs = [(i,j) for (i,j) in ijs if i < j]
    
    candidate_adds = []
    
    for i, j in ijs:
        for x in neighborhood_leq(G, i, 1):
            for y in neighborhood_leq(G, j, 1):
                if y < x:
                    x,y = y,x
                if x != y and not A[x,y] and not (x,y) in candidates:
                    candidates[(x,y)] = mses[mapping[(x,y)]]
        if prioritise_betweenness and len(candidates) > 0:
            break
    
    return candidates


STR_TO_CURVATURE_MAP = {
    'uma_forman': uma_forman_parallel,
}
STR_TO_RICCI_STEP = {
    'fsdrf': focused_stochastic_ricci_step,
}


def stochastic_discrete_ricci_flow(
    A,
    N,
    curvature_fn,
    target_curvature=0,
    scaling=2,
    only_allow_positive_actions=True,
    give_graphs_ricci_attribute=True,
    prioritise_betweenness=False,
    consider_positivity=True,
    ricci_step='fsdrf',
    max_steps=None
):
    G = nx.from_numpy_matrix(A)
    added_edges = []
    removed_edges = []

    if isinstance(curvature_fn, str):
        curvature_fn = STR_TO_CURVATURE_MAP[curvature_fn]

    if isinstance(ricci_step, str):
        ricci_step = STR_TO_RICCI_STEP[ricci_step]
    
    i = max_steps if max_steps is not None else -1
    while i != 0:
        scores = ricci_step(
            G,
            A,
            N,
            curvature_fn,
            target_curvature=target_curvature,
            give_graphs_ricci_attribute=give_graphs_ricci_attribute,
            prioritise_betweenness=prioritise_betweenness,
            consider_positivity=consider_positivity
        )
        scores_keys = list(scores.keys())
        scores_values = np.array(list(scores.values()))
        if np.any(np.array() > 0):
            if only_allow_positive_actions:
                scores = {k:v for k,v in scores.items() if v > 0}
            if len(scores_keys) > 1:
                x,y = scores_keys[
                    np.random.choice(
                        range(len(scores_keys)),
                        p=softmax(scores_values*scaling)
                    )
                ]
            else:
                x,y = scores_keys[0]
            if G.has_edge(x,y):
                G.remove_edge(x,y)
                removed_edges.append((x,y))
            else:
                G.add_edge(x,y)
                added_edges.append((x,y))
            i -= 1
        else:
            break
    
    return added_edges, removed_edges
