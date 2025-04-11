import os
import numpy as np
from natsort import natsorted
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger("[Localizer]") # logger level is explicitly set below by LOG_LEVEL (TODO: Neat up!)

from libs.matcher import lightglue as matcher_lg
from libs.matcher import generic_matcher
from libs.commons.utils import rle_to_mask, nodes2key

from libs.logger.level import LOG_LEVEL
logger.setLevel(LOG_LEVEL)


class LocalizeTopological:
    def __init__(self, imgDir, mapGraph, W, H, mapImgPositions=None, cfg={}):
        self.imgNames = natsorted(os.listdir(f'{imgDir}'))
        self.imgNames = [f'{imgDir}/{imgName}' for imgName in self.imgNames]
        logger.info(
            f"{len(self.imgNames)} images in the map directory {imgDir}")

        self.mapGraph = mapGraph
        self.nodeID_to_imgRegionIdx = np.array(
            [mapGraph.nodes[node]['map'] for node in mapGraph.nodes()])
        self.mapImgPositions = mapImgPositions

        self.map_features_list = None
        self.qry_features = None
        self.matcher_name = cfg.get("matcher_name", "lightglue")
        if self.matcher_name == "lightglue":
            self.matcher = matcher_lg.MatchLightGlue(W, H, cfg=cfg)
            logger.info("Precomputing map features...")
            self.map_features_list = self.precompute_map_features()

        elif self.matcher_name == "sam2":
            self.matcher = cfg["sam2_tracker"]
        else:
            self.matcher = generic_matcher.IMMGeneric(
                resize_w=W, resize_h=H, matcher_name=self.matcher_name, cfg=cfg)

        self.try_relocalize = not cfg.get("use_gt_localization", False)
        self.reloc_dia = 2 * cfg.get("loc_radius", 4)
        self.reloc_dia_add = 5
        self.reloc_dia_max = 30
        self.reloc_dia_curr = self.reloc_dia

        self.subsample_ref = cfg['subsample_ref']
        self.localizer_iter_lb = 0
        self.localizer_iter_ub = len(self.imgNames) if cfg.get(
            "goalImgIdx", None) is None else cfg["goalImgIdx"] + 1
        logger.info(
            f"Localizer iterator bounds are set as: {self.localizer_iter_lb}, {self.localizer_iter_ub}")
        self.localizedImgIdx = 0
        self.greedy_propeller = False
        self.lost = False

    def precompute_map_features(self):
        map_features_list = []
        for imgName in tqdm(self.imgNames):
            image_tensor = self.matcher.getImg(imgName)
            map_features = self.matcher.lexor.extract(image_tensor)
            for key in map_features.keys():
                map_features[key] = map_features[key].detach().cpu().numpy()
            map_features_list.append(map_features)
        return map_features_list

    def update_localizer_iter_lb(self):
        if self.greedy_propeller:
            if self.localizedImgIdx > self.localizer_iter_lb:
                self.localizer_iter_lb = self.localizedImgIdx
        else:
            self.localizer_iter_lb = max(
                0, self.localizedImgIdx - self.reloc_dia//2)

    def getRefImgInds(self):
        lb = self.localizer_iter_lb
        ub = min(self.localizer_iter_lb + self.reloc_dia, self.localizer_iter_ub)
        refImgInds = np.arange(lb, ub)[::self.subsample_ref]
        return refImgInds

    def relocalize(self, qryImg, qryNodes):
        self.reloc_dia_curr += self.reloc_dia_add
        logger.info(f"Relocalizing with reloc_dia: {self.reloc_dia_curr}")
        matchPairs = self.localize(qryImg, qryNodes)
        return matchPairs

    def get_closest_map_image_index(self, qryPosition):
        dists = np.linalg.norm(self.mapImgPositions - qryPosition, axis=1)
        return np.argmin(dists)

    def localize(self, qryImg, qryNodes, qryPosition=None, refImgInds=None):
        if refImgInds is None:
            self.update_localizer_iter_lb()
            refImgInds = self.getRefImgInds()

        refImgList, refNodesList, refNodesIndsList = [], [], []
        for refInd in refImgInds:
            refImgList.append(self.imgNames[refInd])
            refNodesInds = np.argwhere(
                self.nodeID_to_imgRegionIdx[:, 0] == refInd).flatten()
            refNodesIndsList.append(refNodesInds)
            if self.matcher_name == "sam2":
                refNodesList.append([{"segmentation": rle_to_mask(self.mapGraph.nodes[n]["segmentation"]),
                                      "bbox": self.mapGraph.nodes[n]["bbox"]} for n in refNodesInds])
            else:
                refNodesList.append([self.mapGraph.nodes(
                    data=True)[n] for n in refNodesInds])

        if self.matcher_name == "sam2":
            refImgList_np = [cv2.resize(cv2.imread(path)[
                                        :, :, ::-1], (self.matcher.resize_w, self.matcher.resize_h)) for path in refImgList]

            matchPairsList, _ = self.matcher.track_segments_in_sequence(
                qryImg, refImgList_np, qryNodes, refNodesList, query_frame_idx_in_sequence=0)

            # iterate over each ref image's tracked indices
            matchPairs = []
            for i, refIndsLocal in enumerate(matchPairsList.T):
                keep_tracked = refIndsLocal != -1
                matchedRefNodeInds_i = refNodesIndsList[i][refIndsLocal[keep_tracked]]
                qryNodeInds_i = np.arange(len(qryNodes))[keep_tracked]
                matchPairs.append(np.column_stack(
                    [qryNodeInds_i, matchedRefNodeInds_i]))
            if len(matchPairs) > 0:
                matchPairs = np.vstack(matchPairs)
                matchedRefNodeInds = matchPairs[:, 1]
            else:
                matchPairs = np.array([])
                matchedRefNodeInds = np.array([])
        else:
            ftTgtList = None
            if self.map_features_list is not None:
                ftTgtList = [self.map_features_list[ind] for ind in refImgInds]
            matchPairsList, _, self.qry_features = self.matcher.matchPair_imgWithMask_multi(
                qryImg, refImgList, qryNodes, refNodesList, ftTgtList=ftTgtList)

            matchedRefNodeInds = np.concatenate(
                [refNodesIndsList[i][matchPairs[:, 1]] for i, matchPairs in enumerate(matchPairsList)])

            matchPairs = np.column_stack(
                [np.vstack(matchPairsList)[:, 0], matchedRefNodeInds])
        logger.info(f"Num matches: {len(matchPairs)}")

        if len(matchPairs) == 0:
            logger.warning("Lost! No matches found.")
            self.lost = True
        else:
            self.lost = False

        # recursively try to relocalize until un-lost or reloc_dia_max reached
        if self.try_relocalize and self.lost and self.reloc_dia_curr <= self.reloc_dia_max:
            matchPairs = self.relocalize(qryImg, qryNodes)
        else:
            self.reloc_dia_curr = self.reloc_dia

        if not self.lost:
            matchedRefImgInds = self.nodeID_to_imgRegionIdx[matchedRefNodeInds][:, 0]
            bc = np.bincount(matchedRefImgInds)  # ,weights=weights)

            self.localizedImgIdx = bc.argmax()
            logger.info(f"Localized to imgIdx: {self.localizedImgIdx}")

        if qryPosition is not None and self.mapImgPositions is not None:
            closestMapImgIdx = self.get_closest_map_image_index(qryPosition)
            logger.info(f"Closest map imgIdx: {closestMapImgIdx}")

        return matchPairs

    def evaluate(self, matchPairs, qryNodes, qry_instance_ids, visualize=False, qryImg=None):
        qryMasks = nodes2key([qryNodes[n]
                             for n in matchPairs[:, 0]], 'segmentation')
        refNodes = [self.mapGraph.nodes(data=True)[n]
                    for n in matchPairs[:, 1]]
        refMasks = nodes2key(refNodes, 'segmentation')

        refImgInds = self.nodeID_to_imgRegionIdx[matchPairs[:, 1]][:, 0]
        instances2 = []
        for refImgInd in refImgInds:
            instance_ids_ref = np.load(
                f"{self.imgNames[refImgInd].replace('images', 'images_sem').replace('.png', '.npy')}", allow_pickle=True)
            instances2.append(instance_ids_ref)
        instances2 = np.array(instances2)
        distances = evaluate_segment_association(
            qryMasks, refMasks, qry_instance_ids[None, ...].repeat(len(qryMasks), axis=0), instances2)

        if visualize:
            assert qryImg is not None, "need query image (as rgb array) for visualization"
            idx = distances.argmin()
            visualize_matched_mask_pair(qryImg, qryMasks[idx], cv2.imread(
                self.imgNames[refImgInds[idx]])[:, :, ::-1], refMasks[idx])

        return distances


def evaluate_segment_association(masks1, masks2, instances1, instances2, max_id=1000):
    """
    Evaluate segment association between two sets of masks.
    masks*: an array of binary masks*, shape (num_masks, H, W)
    instances*: an array of instance IDs*, shape (num_masks, H, W)
    """
    assert instances1.max() < max_id and instances2.max(
    ) < max_id, f"set {max_id=} to a larger value: {instances1.max()=} {instances2.max()=}"

    instance_ids_masks1 = instances1 * masks1  # (num_masks, H, W)
    instance_ids_masks2 = instances2 * masks2  # (num_masks, H, W)
    distances = []
    for i in range(masks1.shape[0]):
        ids_dist1 = np.bincount(
            instance_ids_masks1[i].flatten(), minlength=max_id)
        ids_dist2 = np.bincount(
            instance_ids_masks2[i].flatten(), minlength=max_id)
        distance = np.linalg.norm(ids_dist1 - ids_dist2)
        pixel_counts = np.maximum(
            1e-6, np.maximum(ids_dist1.sum(), ids_dist2.sum()))
        distance /= pixel_counts
        # overlap = (abs(ids_dist1 - ids_dist2) / np.maximum(1, np.maximum(ids_dist1, ids_dist2))).mean()
        distances.append(distance)
    distances = np.array(distances)
    return distances


def visualize_matched_mask_pair(rgb1, mask1, rgb2, mask2, mask_alpha=0.5, display=False, mask_color=[0, 200, 0]):

    vis1, vis2 = rgb1.copy(), rgb2.copy()

    vis1[mask1] = mask_color
    vis2[mask2] = mask_color
    vis1 = cv2.addWeighted(rgb1, 1-mask_alpha, vis1, mask_alpha, 0)
    vis2 = cv2.addWeighted(rgb2, 1-mask_alpha, vis2, mask_alpha, 0)
    vis = np.hstack([vis1, vis2])

    if display:
        plt.imshow(vis)
        plt.show()
    return vis
