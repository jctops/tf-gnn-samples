import networkx as nx
import numpy as np

softmax = lambda x: np.exp(x)/sum(np.exp(x))

def uma_forman_parallel(G, A, N, target_curvature, return_extra=True):
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

    curvatures = (4 - degrees_reshaped - degrees_reshaped_T + 3*AE2 + squares)*AE

    if return_extra:
        C = curvatures[0]
        minima = np.argwhere(C == np.min(C))
        ijs = [(mini[0],mini[1]) for mini in minima]
        return curvatures, AE, no_of_edges, mapping, ijs
    return curvatures


def slanted_adj(A):
    b = np.zeros((A.shape[0], A.shape[1]*A.shape[1]))
    b[:, ::A.shape[1]+1] = A
    return np.swapaxes(b.reshape(A.shape[0], A.shape[1], A.shape[1]), 0, 1)

def slanted_adj_4D(A):
    b = np.zeros((A.shape[0], A.shape[1], A.shape[2]*A.shape[2]))
    b[:, :, ::A.shape[2]+1] = A
    return np.swapaxes(b.reshape(A.shape[0], A.shape[1], A.shape[2], A.shape[2]), 1, 2)

def linalg_way_lambda(A, N):
    Ax = A.reshape(N,N,-1)
    Ay = A.reshape(N,-1,N)
    T = (A.dot(A * (1 - Ax)) * (1 - slanted_adj(A))) - 1
    four_cycles = T * Ax * (1 - Ay)
    sharps = (four_cycles > 0).sum(axis=0)
    lambdas = four_cycles.max(axis=0)
    lambdas = np.maximum(lambdas, lambdas.T)
    return sharps, lambdas

def linalg_way_lambda_3D(AE, N):
    M = len(AE)
    Ax = AE.reshape(M,N,N,-1)
    Ay = AE.reshape(M,N,-1,N)
    T = (np.einsum('...ij,...kjm->...ikm', AE, np.einsum('...jk,...ijk->...ijk', AE, (1-Ax))) * (1 - slanted_adj_4D(AE))) - 1
    four_cycles = T * Ax * (1 - Ay)
    sharps = (four_cycles > 0).sum(axis=1)
    lambdas = four_cycles.max(axis=1)
    lambdas = np.maximum(lambdas, np.einsum('ijk->ikj', lambdas))
    return sharps, lambdas

def unbiased_forman(A, N):
    degrees = A.sum(axis=0)
    degrees_reshaped = degrees.reshape((-1,N))
    degrees_reshaped_T = degrees.reshape((N,-1))
    degrees_max = np.maximum(degrees_reshaped, degrees_reshaped_T)
    degrees_min = np.maximum(degrees_reshaped_T, degrees_reshaped)
    A2 = np.matmul(A,A)
    sharps, lambdas = linalg_way_lambda(A,N)
    return (np.round((2 / degrees_reshaped) + (2 / degrees_reshaped_T) - 2 + 2*A2/degrees_max + A2/degrees_min \
        + np.nan_to_num((sharps + sharps.T)/(lambdas*degrees_max)),2)) * A

def unbiased_forman_focused(G, A, N, target_curvature, return_extra=True):
    base_no_of_edges = A.sum()
    no_of_edges = np.array([base_no_of_edges])
    
    C = unbiased_forman(A,N)
    minC = np.min(C)
    if minC >= target_curvature:
        if return_extra:
            return (C*A).reshape(1,N,N), A.reshape(1,N,N), no_of_edges, {}, []
        else:
            return (C*A).reshape(1,N,N)
    
    minima = np.argwhere(C == minC)
    ijs = [(minimum[0],minimum[1]) for minimum in minima]
    ijs = [(i,j) for (i,j) in ijs if i < j]
    E = np.zeros((1,N,N))
    l = 1
    mapping = {}
    
    for i, j in ijs:
        for x in list(G.neighbors(i)) + [i]:
            for y in list(G.neighbors(j)) + [j]:
                x_, y_ = (x,y) if x<y else (y,x)
                if x_ != y_ and not A[x_,y_] and not (x_,y_) in mapping:
                    E = np.append(E, np.zeros((1,N,N)), axis=0)
                    E[l,x,y] = E[l,y,x] = 1
                    no_of_edges = np.append(no_of_edges, [base_no_of_edges+1])
                    mapping[(x_,y_)] = l
                    l += 1
    NC2 = l
    
    AE = A+E
    AE2 = np.matmul(AE,AE)
    degrees = np.diagonal(AE2, axis1=1, axis2=2)
    degrees_reshaped = degrees.reshape((NC2,-1,N))
    degrees_reshaped_T = degrees.reshape((NC2,N,-1))
    degrees_max = np.maximum(degrees_reshaped, degrees_reshaped_T)
    degrees_min = np.minimum(degrees_reshaped, degrees_reshaped_T)
    sharps, lambdas = linalg_way_lambda_3D(AE,N)
    curvatures = (2 / degrees_reshaped) + (2 / degrees_reshaped_T) - 2 + 2*AE2/degrees_max + AE2/degrees_min + np.nan_to_num((sharps + np.einsum('ijk->ikj', sharps))/(lambdas*degrees_max))
    if return_extra:
        return curvatures*AE, AE, no_of_edges, mapping, ijs
    return curvatures*AE


def focused_stochastic_ricci_step(G, A, N, curvature_fn, target_curvature=0, give_graphs_ricci_attribute=True,
                          prioritise_betweenness=False, consider_positivity=True):
    curvatures, AE, no_of_edges, mapping, ijs = curvature_fn(G, A, N, target_curvature)
    
    candidates = {}
    x = AE*target_curvature - curvatures
    if not consider_positivity:
        x = (abs(x) + x) / 2
    mses = (x**2).sum(axis=2).sum(axis=1)/no_of_edges
    mses = mses[0] - mses
    
    if prioritise_betweenness:
        betweenness = {k:v for (k,v) in nx.algorithms.centrality.edge_betweenness_centrality(G).items() if k in ijs}
        betweenness = {k: v for k, v in sorted(betweenness.items(), key=lambda item: -item[1])}
        ijs = list(betweenness.keys())
    else:
        ijs = [(i,j) for (i,j) in ijs if i < j]
    
    for i, j in ijs:
        for x in list(G.neighbors(i)) + [i]:
            for y in list(G.neighbors(j)) + [j]:
                x_, y_ = (x,y) if x<y else (y,x)
                if x_ != y_ and not A[x_,y_] and not (x_,y_) in candidates and mses[mapping[(x_,y_)]] > 0:
                        candidates[(x_,y_)] = mses[mapping[(x_,y_)]]
        if prioritise_betweenness and len(candidates) > 0:
            break
            
    return candidates


STR_TO_CURVATURE_MAP = {
    'uma_forman': uma_forman_parallel,
    'unbiased_forman': unbiased_forman_focused,
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
    prioritise_betweenness = (prioritise_betweenness==True) or (prioritise_betweenness=="True")
    consider_positivity = (consider_positivity==True) or (consider_positivity=="True")
    G = nx.from_numpy_matrix(A)
    A = A.copy()
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
        if only_allow_positive_actions:
            scores = {k:v*scaling for k,v in scores.items() if v > 0}
        scores_keys = list(scores.keys())
        scores_values = np.array(list(scores.values()))
        if np.any(scores_values > 0):
            if len(scores_keys) > 1:
                x,y = scores_keys[
                    np.random.choice(
                        range(len(scores_keys)),
                        p=softmax(scores_values)
                    )
                ]
            else:
                x,y = scores_keys[0]
            if G.has_edge(x,y):
                G.remove_edge(x,y)
                A[x,y] = A[y,x] = 0
                removed_edges.append((x,y))
            else:
                G.add_edge(x,y)
                A[x,y] = A[y,x] = 1
                added_edges.append((x,y))
            i -= 1
        else:
            break
    
    return added_edges, removed_edges
