import numpy as np
import cv2
from libs.commons import utils

def get_goalNodeIdx(mapPath, G, nodeID_to_imgRegionIdx):
    """
    returns the node index of the segment in the map graph that has the highest IoU with the goal object in the last image of the episode's teach run
    """
    agent_states = np.load(f"{mapPath}/agent_states.npy", allow_pickle=True)
    goalImgIdx = len(agent_states) - 1
    episode = np.load(f"{mapPath}/episode.npy", allow_pickle=True)[()]
    obs_g = np.load(f"{mapPath}/obs_g.npy", allow_pickle=True)[()]
    goalMaskBinary = obs_g['semantic_sensor']==int(episode.goal_object_id)
    mapNodeInds_in_goalImg = np.argwhere(nodeID_to_imgRegionIdx[:,0] == goalImgIdx).flatten()
    mapMasks = utils.nodes2key(mapNodeInds_in_goalImg,'segmentation', G).transpose(1,2,0)
    img_h, img_w, _ = mapMasks.shape
    goalMaskBinary = cv2.resize(goalMaskBinary.astype(float), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    mask_and = np.logical_and(mapMasks, goalMaskBinary[:,:,None])
    mask_or = np.logical_or(mapMasks, goalMaskBinary[:,:,None])
    iou = mask_and.sum(0).sum(0) / mask_or.sum(0).sum(0)
    goalNodeIdx = mapNodeInds_in_goalImg[iou.argmax()]
    return goalNodeIdx