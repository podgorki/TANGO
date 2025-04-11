import numpy as np
import networkx as nx
import logging

logger = logging.getLogger("[Planner]") # logger level is explicitly set below by LOG_LEVEL (TODO: Neat up!)

from libs.logger.level import LOG_LEVEL
from libs.commons.utils import count_edges_with_given_weight, get_edge_weight_types
logger.setLevel(LOG_LEVEL)


class PlanTopological:
    def __init__(self, mapGraph, goalNodeIdx, cfg={}):

        self.mapGraph = mapGraph
        self.nodeID_to_imgRegionIdx = np.array(
            [mapGraph.nodes[node]['map'] for node in mapGraph.nodes()])
        self.cfg = cfg

        self.use_goal_nbrs = self.cfg["use_goal_nbrs"]
        self.plan_da_nbrs = self.cfg["plan_da_nbrs"]
        self.edge_weight_str = self.cfg['edge_weight_str']
        self.precomputed_allPathLengths_found = False
        self.preplan_to_goals_only = False
        self.allPathLengths = mapGraph.graph.get('allPathLengths', {})

        # get goal node neighbors from the same img
        if self.use_goal_nbrs:
            self.goalNodeNbrs = list(self.mapGraph.neighbors(goalNodeIdx))
            goalImgIdx = self.nodeID_to_imgRegionIdx[goalNodeIdx][0]
            self.goalNodeNbrs = [
                n for n in self.goalNodeNbrs if self.nodeID_to_imgRegionIdx[n][0] == goalImgIdx]
        else:
            self.goalNodeNbrs = [goalNodeIdx]
        self.goalNodeNbrsImgIdx = self.nodeID_to_imgRegionIdx[self.goalNodeNbrs, 0]

        if self.edge_weight_str not in self.allPathLengths:
            logger.info(
                f"Path lengths not found in graph, computing topological paths to goal using {self.edge_weight_str=}")
            if self.preplan_to_goals_only:
                self.allPathLengths = np.array([self.get_path(
                    None, g, self.mapGraph, weight=self.edge_weight_str, all2tgt=True) for g in self.goalNodeNbrs]).T
            else:
                self.allPathLengths = self.get_path(
                    None, None, self.mapGraph, weight=self.edge_weight_str, allPairs=True)
            logger.info("Done computing path lengths.")
        else:
            self.precomputed_allPathLengths_found = True
            logger.info(
                f"Path lengths found in graph, using {self.edge_weight_str=}")
            self.allPathLengths = self.allPathLengths[self.edge_weight_str]

        if self.plan_da_nbrs:
            logger.info("Precomputing DA nbrs")
            self.daNbrs = self.precompute_nbrs(self.mapGraph, edgeType='da')

    def precompute_nbrs(self, G, edgeType='da'):
        nbrs = []
        for node in G.nodes():
            nbrs.append([n for n in G.neighbors(node)
                        if G.edges[node, n].get('edgeType') == edgeType])
        return nbrs

    def get_path(self, src, tgt, G, weight=None, allPairs=False, all2tgt=False):

        if count_edges_with_given_weight(G, weight) == 0:
            raise ValueError(
                f'No edges found for given {weight=}, found {get_edge_weight_types(G)=}')

        if allPairs or all2tgt:
            # this returns lengths
            if all2tgt:
                pathLengths = dict(nx.shortest_path_length(
                    G, target=tgt, weight=weight))
                pathLengths = np.array(
                    [pathLengths.get(src, 1e6) for src in G.nodes()])
            else:
                pathLengths = dict(
                    nx.all_pairs_dijkstra_path_length(G, weight=weight))
                pathLengths = np.array(
                    [[pathLengths[src].get(tgt, 1e6) for tgt in G.nodes()] for src in G.nodes()])
            pathLengths = np.nan_to_num(
                pathLengths, nan=1e6, posinf=1e6, neginf=1e6)
            return pathLengths
        else:
            # this returns paths
            shortest_path = nx.shortest_path(
                G, source=src, target=tgt, weight=weight)
            return shortest_path

    def get_pathLengths_matchedNodes(self, matchedRefNodeInds):
        meanPathLengths = []
        # fast versions possible too
        # pl = self.allPathLengths[self.daNbrs[matchPairs[:,1]],self.goalNodeNbrs].min(-1)
        # pl = self.allPathLengths[matchPairs[:,1],self.goalNodeNbrs].min(-1)
        pl = []
        nodesClose2Goal = []
        for s in matchedRefNodeInds:
            # p = np.min([len(self.get_path(s,g,weight=self.edge_weight_str)) for g in self.goalNodeNbrs])
            # p = np.min([self.allPathLengths[s,g] for g in self.goalNodeNbrs])
            plMinPerMatchNbr = []
            s_nbrs = [s]
            # minimize path length over DA nbrs of the matched ref node
            if self.plan_da_nbrs:
                s_nbrs += self.daNbrs[s]

            for s2 in s_nbrs:
                if self.preplan_to_goals_only:
                    plMinPerMatchNbr.append(
                        np.min([self.allPathLengths[s2, gi] for gi in range(len(self.goalNodeNbrs))]))
                else:
                    plMinPerMatchNbr.append(
                        np.min([self.allPathLengths[s2, g] for g in self.goalNodeNbrs]))

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

        # log without the 1e6 values
        pmin, pmed, pavg, pmax = pl.min(), np.median(pl), pl.mean(), pl.max()
        inliers = pl != 1e6
        if len(inliers) != 0:
            pmin, pmed, pavg, pmax = pl[inliers].min(), np.median(pl[inliers]), pl[inliers].mean(), pl[inliers].max()
        meanPathLengths.append(pavg)
        logger.info(
            f"Path length mean: {pavg:.2f}, median: {pmed:.2f}, min: {pmin:.2f}, max: {pmax:.2f}")
        return pl, nodesClose2Goal
