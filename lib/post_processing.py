from itertools import chain

import numpy as np
from networkx import Graph, adjacency_matrix
from scipy.sparse import diags
from sklearn.neighbors import NearestNeighbors


def make_graph(coords: np.array, radius=9):
    num_patches = len(coords)

    nn = NearestNeighbors(n_neighbors=radius)
    nn.fit(coords)
    _, neigh = nn.kneighbors(coords)

    a = np.repeat(range(num_patches), radius-1)
    b = np.fromiter(chain(*[neigh[v_idx][1:]
                            for v_idx in range(num_patches)]
                    ),dtype=int)

    edge_spatial = np.stack([a,b])
    G = Graph()
    G.add_nodes_from(coords)
    G.add_edges_from([(tuple(coords[n1]), tuple(coords[n2])) for n1, n2 in edge_spatial.T])

    return G


def propagate_scores(graph, scores, alpha, steps, threshold):
    adj = adjacency_matrix(graph, weight=None)
    D = np.array(adj.sum(axis=1)).flatten()
    degree = diags(1 / np.sqrt(D), offsets=0)
    S = degree @ adj @ degree

    c, a, scores = zip(*scores)
    scores = np.array(scores)
    masked_scores = np.ones_like(scores)
    masked_scores[scores < threshold] = 0

    residual = (1 - alpha) * scores
    out = masked_scores

    # Confidence propagation / Label smoothing
    for _ in range(steps):
        out = alpha * (S @ out)

    out = residual + out

    pp_scores = list(zip(c,a,out))

    return pp_scores
