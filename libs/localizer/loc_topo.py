import os
import numpy as np
from natsort import natsorted

import logging

logger = logging.getLogger("[Localizer]")

from libs.matcher import lightglue as matcher_lg


class Localize_Topological:
    def __init__(self, imgDir, mapGraph, W, H, mapImgPositions=None):
        self.imgNames = natsorted(os.listdir(f'{imgDir}'))
        self.imgNames = [f'{imgDir}/{imgName}' for imgName in self.imgNames]
        print(f"{len(self.imgNames)} images in the map directory {imgDir}")

        self.mapGraph = mapGraph
        self.nodeID_to_imgRegionIdx = np.array([mapGraph.nodes[node]['map'] for node in mapGraph.nodes()])
        self.mapImgPositions = mapImgPositions

        self.matcher = matcher_lg.MatchLightGlue(W, H)

        self.reloc_rad = 5
        self.reloc_rad_add = 5
        self.reloc_rad_max = 30
        self.reloc_rad_curr = self.reloc_rad

        self.localizer_iter_lb = 0
        self.localizedImgIdx = 0
        self.greedy_propeller = False
        self.lost = False

    def update_localizer_iter_lb(self):
        if self.greedy_propeller:
            if self.localizedImgIdx > self.localizer_iter_lb:
                self.localizer_iter_lb = self.localizedImgIdx
        else:
            self.localizer_iter_lb = max(0, self.localizedImgIdx - self.reloc_rad // 2)

    def getRefImgInds(self):
        return np.arange(self.localizer_iter_lb, min(self.localizer_iter_lb + self.reloc_rad, len(self.imgNames)))

    def relocalize(self, qryImg, qryNodes):
        self.reloc_rad_curr += self.reloc_rad_add
        logger.info(f"Relocalizing with reloc_rad: {self.reloc_rad_curr}")
        matchPairs = self.localize(qryImg, qryNodes)
        return matchPairs

    # def relocalize(self, qryImg, qryNodes):
    #     reloc_rad_orig = self.reloc_rad
    #     attempts = 0
    #     while self.reloc_rad <= self.reloc_rad_max and self.lost:
    #         self.expand_reloc_rad()
    #         logger.info(f"Relocalizing with reloc_rad: {self.reloc_rad}")
    #         matchPairs = self.localize(qryImg, qryNodes)
    #         attempts += 1
    #     if not self.lost:
    #         logger.info(f"Relocalized after {attempts} attempts with reloc_rad: {self.reloc_rad}")
    #         self.lost = False
    #     else:
    #         logger.warning(f"Relocalization failed after {attempts} attempts up to reloc_rad: {self.reloc_rad}")
    #     self.reloc_rad = reloc_rad_orig
    #     return matchPairs

    def get_closest_map_image_index(self, qryPosition):
        dists = np.linalg.norm(self.mapImgPositions - qryPosition, axis=1)
        return np.argmin(dists)

    def localize(self, qryImg, qryNodes, qryPosition=None):
        self.update_localizer_iter_lb()
        refImgInds = self.getRefImgInds()

        refImgList, refNodesList, refNodesIndsList = [], [], []
        for refInd in refImgInds:
            refImgList.append(self.imgNames[refInd])
            refNodesInds = np.argwhere(self.nodeID_to_imgRegionIdx[:, 0] == refInd).flatten()
            refNodesIndsList.append(refNodesInds)
            refNodesList.append([self.mapGraph.nodes(data=True)[n] for n in refNodesInds])

        matchPairsList, _ = self.matcher.matchPair_imgWithMask_multi(qryImg, refImgList, qryNodes, refNodesList)

        matchedRefNodeInds = np.concatenate(
            [refNodesIndsList[i][matchPairs[:, 1]] for i, matchPairs in enumerate(matchPairsList)])

        matchPairs = np.column_stack([np.vstack(matchPairsList)[:, 0], matchedRefNodeInds])
        logger.info(f"Num matches: {len(matchPairs)}")

        if len(matchPairs) == 0:
            logger.warning("Lost! No matches found.")
            self.lost = True
        else:
            self.lost = False

        # recursively try to relocalize until un-lost or reloc_rad_max reached
        if self.lost and self.reloc_rad_curr <= self.reloc_rad_max:
            matchPairs = self.relocalize(qryImg, qryNodes)
        else:
            self.reloc_rad_curr = self.reloc_rad

        if not self.lost:
            matchedRefImgInds = self.nodeID_to_imgRegionIdx[matchedRefNodeInds][:, 0]
            bc = np.bincount(matchedRefImgInds)  # ,weights=weights)

            self.localizedImgIdx = bc.argmax()
            logger.info(f"Localized to imgIdx: {self.localizedImgIdx}")

        if qryPosition is not None and self.mapImgPositions is not None:
            closestMapImgIdx = self.get_closest_map_image_index(qryPosition)
            logger.info(f"Closest map imgIdx: {closestMapImgIdx}")

        return matchPairs