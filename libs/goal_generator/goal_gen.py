import os
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import time
import sys
from typing import Optional

import logging

logger = logging.getLogger("[Goal Generator]")

from libs.localizer import loc_topo
from libs.planner_global import plan_topo
from libs.commons import utils, utils_data, utils_viz


class Goal_Gen:

    def __init__(self, W: int, H: int, G, map_path, poses: Optional = None, ):
        self.W = W
        self.H = H
        self.max_pl = 100
        self.qIter = -1
        self.goalMask_default = self.max_pl * np.ones((self.H, self.W))  # default value for invalid goal segments
        self.G = G
        self.map_path = map_path
        self.positions = None
        if poses is not None:
            self.positions = np.array([pose.position for pose in poses])

        utils.change_edge_attr(self.G)
        self.nodeID_to_imgRegionIdx = np.array([self.G.nodes[node]['map'] for node in self.G.nodes()])

        self.goalNodeIdx = utils_data.get_goalNodeIdx(self.map_path, self.G, self.nodeID_to_imgRegionIdx)
        self.localizer = loc_topo.Localize_Topological(
            f"{self.map_path}/images", self.G, self.W, self.H, mapImgPositions=self.positions
        )
        self.planner_g = plan_topo.PlanTopological(self.G, self.goalNodeIdx)

    def get_goal_mask(self, qryImg, qryNodes, qryPosition: Optional = None):
        self.qIter += 1
        logger.info(f"Iter: {self.qIter}")
        printTime = False
        t1 = time.time()
        # self.qryNodes = self.segmentor.segment(qryImg)
        self.qryNodes = qryNodes

        if printTime: print(f"Segmentation time: {time.time() - t1:.2f}s")

        if self.qryNodes is None:
            self.localizer.lost = True
        else:
            self.qryMasks = utils.nodes2key(self.qryNodes, 'segmentation')
            self.qryCoords = utils.nodes2key(self.qryNodes, 'coords')
            if printTime: print(f"Seg Proc time: {time.time() - t1:.2f}s")

            self.matchPairs = self.localizer.localize(qryImg, self.qryNodes, qryPosition)

        if self.localizer.lost:
            # let controller enter explore mode (return default goal mask)?
            # resume localization from the new observation
            self.localizer.lost = False
            self.pls = self.max_pl * np.ones(len(self.qryCoords))
            self.coords = self.qryCoords
            return self.goalMask_default
        if printTime: print(f"Localization time: {time.time() - t1:.2f}s")

        self.pls, nodesClose2Goal = self.planner_g.get_pathLengths_matchedNodes(self.matchPairs[:, 1])
        self.coords = self.qryCoords[self.matchPairs[:, 0]]

        if printTime: print(f"Planning time: {time.time() - t1:.2f}s")

        self.goalMask = self.goalMask_default.copy()
        for i in range(len(self.pls)):
            self.goalMask[self.qryMasks[self.matchPairs[i, 0]]] = self.pls[i]
        if printTime: print(f"Goal mask time: {time.time() - t1:.2f}s")

        return self.goalMask

    def visualize_goal_mask(self, qryImg, display: bool = False):
        colors, norm = utils_viz.value2color(self.pls, cmName='viridis')
        vizImg = utils_viz.drawMasksWithColors(qryImg, self.qryMasks[self.matchPairs[:, 0]], colors)
        if display:
            plt.imshow(vizImg)
            plt.colorbar()
            plt.show()
        return vizImg


# Example usage
# python -m libs.goal_generator.goal_gen /path/to/map/ /path/to/model/ (optional)
if __name__ == "__main__":
    # modelPath = f"{os.path.expanduser('~')}/workspace/s/sg_habitat/models/segment-anything/"
    # mapPath = f"{os.path.expanduser('~')}/fastdata/navigation/hm3d_iin_train/1S7LAXRdDqK_0000000_plant_42_/"

    mapPath = sys.argv[1]
    if len(sys.argv) > 2:
        segmentor = "sam"
        modelPath = sys.argv[2]
    else:
        segmentor = "fast_sam"
        modelPath = None

    goalie = Goal_Gen(modelPath, W=320, H=240, segmentor=segmentor)
    goalie.load_episode(mapPath)

    qryImg = cv2.imread(f"{mapPath}/images/00000.png")[:, :, ::-1]
    qryImg = cv2.resize(qryImg, (goalie.W, goalie.H))

    goalMask = goalie.get_goal_mask(qryImg)
    _ = goalie.visualize_goal_mask(qryImg, display=True)
