import numpy as np
import networkx as nx
import os
import logging

logger = logging.getLogger("[Planner]")


class Plan_Topological:
    def __init__(self, mapGraph, goalNodeIdx):

        self.mapNodeWeightStr = "margin"
        self.mapGraph = mapGraph
        self.nodeID_to_imgRegionIdx = np.array([mapGraph.nodes[node]['map'] for node in mapGraph.nodes()])
        self.allPathLengths = mapGraph.graph['allPathLengths']
        # get goal node neighbors from the same img
        goalNodeNbrs = list(self.mapGraph.neighbors(goalNodeIdx))
        goalImgIdx = self.nodeID_to_imgRegionIdx[goalNodeIdx][0]
        self.goalNodeNbrs = [n for n in goalNodeNbrs if self.nodeID_to_imgRegionIdx[n][0] == goalImgIdx]
        self.goalNodeNbrsImgIdx = self.nodeID_to_imgRegionIdx[self.goalNodeNbrs, 0]

        self.plan_da_nbrs = True
        if self.plan_da_nbrs:
            logger.info("Precomputing DA nbrs")
            self.daNbrs = self.precompute_nbrs(self.mapGraph, edgeType='da')

    def precompute_nbrs(self, G, edgeType='da'):
        nbrs = []
        for node in G.nodes():
            nbrs.append([n for n in G.neighbors(node) if G.edges[node, n].get('edgeType') == edgeType])
        return nbrs

    def get_path(self, src, tgt, G=None, weight=None, allPairs=False):
        if G is None:
            if self.G4 is not None:
                G = self.G4
            elif self.G3_ is not None:
                G = self.G3_
                src = self.G3.get_contracted_node(src)
                tgt = self.G3.get_contracted_node(tgt)
            else:
                G = self.G
        if allPairs:
            # this returns lengths
            pathLengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
            pathLengths = np.array([[pathLengths[src][tgt] for tgt in G.nodes()] for src in G.nodes()])
            return pathLengths
        else:
            # this returns paths
            shortest_path = nx.shortest_path(G, source=src, target=tgt, weight=weight)
            return shortest_path

    def get_pathLengths_matchedNodes(self, matchedRefNodeInds):
        meanPathLengths = []
        ## fast versions possible too
        # pl = self.allPathLengths[self.daNbrs[matchPairs[:,1]],self.goalNodeNbrs].min(-1)
        # pl = self.allPathLengths[matchPairs[:,1],self.goalNodeNbrs].min(-1)
        pl = []
        nodesClose2Goal = []
        for s in matchedRefNodeInds:
            # p = np.min([len(self.get_path(s,g,weight=self.mapNodeWeightStr)) for g in self.goalNodeNbrs])
            # p = np.min([self.allPathLengths[s,g] for g in self.goalNodeNbrs])
            plMinPerMatchNbr = []
            s_nbrs = [s]
            # minimize path length over DA nbrs of the matched ref node
            if self.plan_da_nbrs:
                s_nbrs += self.daNbrs[s]

            for s2 in s_nbrs:
                plMinPerMatchNbr.append(np.min([self.allPathLengths[s2, g] for g in self.goalNodeNbrs]))

            if len(plMinPerMatchNbr) == 0:  # POSSIBLE BUG
                p = -1
                nbrClosest2goal = s
            else:
                p = np.min(plMinPerMatchNbr)
                nbrClosest2goal = s_nbrs[np.argmin(plMinPerMatchNbr)]
            nodesClose2Goal.append(nbrClosest2goal)
            pl.append(p)
        pl = np.array(pl)
        pl[pl == -1] = pl.max() + 1
        meanPathLengths.append(pl.mean())
        logger.info(f"Path length mean: {pl.mean():.2f}, median: {np.median(pl):.2f}, min: {pl.min()}, max: {pl.max()}")
        return pl, nodesClose2Goal