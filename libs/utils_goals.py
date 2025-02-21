import numpy as np
import networkx as nx
import torch
from typing import Tuple


def change_edge_attr(G):
    for e in G.edges(data=True):
        if 'margin' in e[2]:
            e[2]['margin'] = 0.0
    return G


def find_graph_instance_ids_and_path_lengths(G4, goal_object_id, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    G_insta_ids = np.array([G4.nodes[n]['instance_id'] for n in G4.nodes])
    goalNodeIdx = np.argwhere(G_insta_ids == goal_object_id).flatten()[-1]
    plsDict = nx.single_source_dijkstra_path_length(G4, goalNodeIdx, weight='margin')
    pls = np.array([plsDict[n] for n in range(len(G4.nodes))])
    G_insta_ids = torch.from_numpy(G_insta_ids.astype(np.int16)).to(device)
    pls = torch.from_numpy(pls.astype(np.int16)).to(device)
    return G_insta_ids, pls


def getGoalMask(G_insta_ids: torch.Tensor, pls: torch.Tensor, sem: np.ndarray, device: str = 'cpu') -> np.ndarray:
    sem = torch.from_numpy(sem).to(device)
    matches = sem[:, :, None] == G_insta_ids[None, None, :]
    matchedNodeInds = torch.argmax(matches.to(int), 2)
    matchedNodeInds_pls = pls[matchedNodeInds]

    semCounts = torch.bincount(sem.flatten())
    semInvalid = semCounts[sem] < 10  # small segments
    matchedNodeInds_pls[semInvalid] = 100  # assign high pl
    matchesInvalid = torch.sum(matches, 2) == 0  # no match
    matchedNodeInds_pls[matchesInvalid] = 101  # assign high pl
    return matchedNodeInds_pls.cpu().numpy()