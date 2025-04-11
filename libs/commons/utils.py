import numpy as np
import networkx as nx
from typing import List, Dict, Any


def count_edges_with_given_weight(G, edge_weight_str):
    if edge_weight_str is None:
        return len(G.edges())
    return sum([1 for e in G.edges(data=True) if e[2].get(edge_weight_str) is not None])


def get_edge_weight_types(G):
    edge_weight_types = set()
    for e in G.edges(data=True):
        for k in e[2].keys():
            edge_weight_types.add(k)
    return edge_weight_types


def change_edge_attr(G):
    for e in G.edges(data=True):
        if 'margin' in e[2]:
            e[2]['margin'] = 0.0
    return G


def norm_minmax(costs, max_val=1):
    costs = costs - costs.min()
    if costs.max() != 0:
        costs = costs / costs.max()
    return (costs * max_val)


def normalize_pls(pls, scale_factor=100, outlier_value=99):
    # remove outlier values if exist
    if pls.max() >= outlier_value:
        # if all are outliers, set them to zero
        if pls.min() >= outlier_value:
            pls = np.zeros_like(pls)
            return pls
        # else set outliers to max value of inliers + 1
        # so that when normalized, they are set to 0
        else:
            pls[pls >= outlier_value] = pls[pls < outlier_value].max() + 1
            # include a dummy value to ensure that the size is the same as the below case
            pls = np.concatenate([pls, [pls.max()]])
    # no outliers
    else:
        # include a dummy value to ensure that the max value is same as that with outliers
        pls = np.concatenate([pls, [pls.max() + 1]])

    # normalize so that outliers are set to 0
    # inliers are ranged (0, scale_factor]
    pls = scale_factor * (pls.max() - pls) / (pls.max() - pls.min())
    return pls[:-1]


def modify_graph(G,nodes,edges):
    G2 = nx.Graph()
    G2.add_nodes_from(nodes)
    G2.add_edges_from(edges)
    G2.graph = G.graph.copy()
    print("Number of nodes & edges in G: ", len(G.nodes), len(G.edges))
    print("Number of nodes & edges in G2: ", len(G2.nodes), len(G2.edges))
    print(f"is_connected(G): {nx.is_connected(G)}")
    print(f"is_connected(G2): {nx.is_connected(G2)}")
    return G2

def intersect_tuples(a, b):
    # Convert lists of tuples to structured arrays
    a_arr = np.array(a, dtype=[('f1', 'int64'), ('f2', 'int64')])
    b_arr = np.array(b, dtype=[('f1', 'int64'), ('f2', 'int64')])

    # Find the intersection
    intersection = np.intersect1d(a_arr, b_arr)

    # Convert the structured arrays back to list of tuples
    return [tuple(row) for row in intersection]

def getSplitEdgeLists(G,flipSim=True):
    if not flipSim:
        raise NotImplementedError
    intraImage_edges = [e for e in G.edges(data=True) if 'sim' not in e[2]]
    da_edges = [(e[0],e[1],{'sim':1-e[2]['sim']}) for e in G.edges(data=True) if 'sim' in e[2]]
    temporal_edges = [(e[0],e[1],{'sim':1-e[2]['sim']}) for e in G.graph['temporalEdges']]

    # find intersection between da_edges and temporal_edges
    da_edges_noAttr = [tuple(sorted((e[0],e[1]))) for e in da_edges]
    temporal_edges_noAttr = [tuple(sorted((e[0],e[1]))) for e in temporal_edges]
    intersection = intersect_tuples(da_edges_noAttr,temporal_edges_noAttr)
    numCommon = len(intersection)

    print(f"Number of intraImage_edges: {len(intraImage_edges)}")
    print(f"Number of da_edges: {len(da_edges)}")
    print(f"Number of temporal_edges: {len(temporal_edges)}")
    print(f"Number of non-intersecting edges (ideally 0): {len(temporal_edges)-numCommon}")

    return intraImage_edges, da_edges, temporal_edges

def mask_to_rle_numpy(array: np.ndarray) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = array.shape
    array = np.transpose(array, (0, 2, 1)).reshape(b, -1)

    # Compute change indices
    diff = array[:, 1:] != array[:, :-1]
    change_indices = np.nonzero(diff)

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[1][change_indices[0] == i]
        cur_idxs = np.concatenate(
            [
                np.array([0], dtype=cur_idxs.dtype),
                cur_idxs + 1,
                np.array([h * w], dtype=cur_idxs.dtype),
            ]
        )
        btw_idxs = np.diff(cur_idxs)
        counts = [] if array[i, 0] == 0 else [0]
        counts.extend(btw_idxs.tolist())
        out.append({"size": [h, w], "counts": counts})
    return out

def rle_to_mask(rle) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order

def nodes2key(nodeInds, key, G=None):
    _key = key
    if key == 'coords':
        _key = 'segmentation'
    if isinstance(nodeInds[0],dict):
        if _key == 'segmentation' and type(nodeInds[0][_key]) == dict:
            values = np.array([rle_to_mask(n[_key]) for n in nodeInds])
        else:
            values = np.array([n[_key] for n in nodeInds])
    else:
        assert G is not None, "nodes can either be dict or indices of nx.Graph"
        if _key == 'segmentation' and type(G.nodes[nodeInds[0]][_key]) == dict:
            values = np.array([rle_to_mask(G.nodes[n][_key]) for n in nodeInds])
        else:
            values = np.array([G.nodes[n][_key] for n in nodeInds])
    if key == 'coords':
        values = np.array([np.array(np.nonzero(v)).mean(1)[::-1].astype(int) for v in values])
    return values