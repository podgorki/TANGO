import numpy as np
import networkx as nx
import torch
from typing import Tuple

import warnings
# python 3.10
warnings.filterwarnings("ignore", category=UserWarning, module="torch.masked.maskedtensor.creation")
# python 3.9
warnings.filterwarnings("ignore", category=UserWarning, module="torch.masked.maskedtensor.core")

from libs.commons import utils

def change_edge_attr(G):
    for e in G.edges(data=True):
        if 'margin' in e[2]:
            e[2]['margin'] = 0.0
    return G


def find_graph_instance_ids_and_path_lengths(G4, goal_object_id, device: str = 'cpu', weight: str = 'margin') -> Tuple[torch.Tensor, torch.Tensor]:
    max_val = 255
    G_insta_ids = np.array([G4.nodes[n]['instance_id'] for n in G4.nodes])
    goalNodeIdx = np.argwhere(G_insta_ids == goal_object_id).flatten()[-1]

    if utils.count_edges_with_given_weight(G4, weight) == 0:
        raise ValueError(f'No edges found for given {weight=}, found {utils.get_edge_weight_types(G4)=}')

    plsDict = nx.single_source_dijkstra_path_length(G4, goalNodeIdx, weight=weight)
    pls = np.array([plsDict.get(n, max_val) for n in range(len(G4.nodes))])
    # check nan
    if np.isnan(pls).any() or np.isinf(pls).any():
        pls = np.nan_to_num(pls, nan=max_val, posinf=max_val, neginf=max_val)
    G_insta_ids = torch.from_numpy(G_insta_ids.astype(np.int16)).to(device)
    if weight is not None and ('geodesic' in weight or 'e3d' in weight):
        pass
    else:
        pls = pls.astype(np.int16) # is this needed anymore? if so, as uint8?
    pls = torch.from_numpy(pls).to(device)
    return G_insta_ids, pls


def getGoalMask(G_insta_ids: torch.Tensor, pls: torch.Tensor, sem: np.ndarray, device: str = 'cpu') -> np.ndarray:

    max_val = 255
    sem = torch.from_numpy(sem).to(device)
    method = 'new'
    if method == 'old':
        matches = sem[:, :, None] == G_insta_ids[None, None, :]
        matchedNodeInds = torch.argmax(matches.to(int), 2)
        matchedNodeInds_pls = pls[matchedNodeInds]
    elif method == 'new':
        curr_instance_ids = torch.unique(sem) # N_c
        # find instance matches from curr to graph
        matches = curr_instance_ids[:, None] == G_insta_ids[None, :] # N_c x N_g

        # find the minimum path length for each instance
        matches_pls = torch.masked.masked_tensor(pls.repeat(matches.shape[0], 1), matches) # N_c x N_g
        matches_pls_min = torch.masked.amin(matches_pls, 1) # N_c
        matches_pls_min = torch.nan_to_num(matches_pls_min, nan=max_val, posinf=max_val, neginf=max_val)

        # create an array of path lengths as per instance ids
        masks_bin = sem[:, :, None] == curr_instance_ids[None, None, :] # H x W x N_c
        matchedNodeInds_pls = (masks_bin * matches_pls_min).sum(-1) # H x W

    return matchedNodeInds_pls.cpu().numpy()
