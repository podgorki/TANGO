import os
import cv2
import numpy as np


def get_goal_info(map_path):
    episode_data_path = f"{map_path}/episode.npy"
    if os.path.exists(episode_data_path):
        episode = np.load(f"{map_path}/episode.npy", allow_pickle=True)[()]
        goal_instance_id = episode.goal_object_id
    else:
        goal_instance_id = int(map_path.split('_')[-2])

    goal_img_index = len(os.listdir(f"{map_path}/images/")) - 1
    instance_mask = np.load(f"{map_path}/obs_g.npy", allow_pickle=True)[()]['semantic_sensor']

    return goal_img_index, instance_mask, goal_instance_id


def match_map_masks_with_goal_mask(map_masks, goal_mask_binary):
    img_h, img_w, _ = map_masks.shape
    goal_mask_binary = cv2.resize(goal_mask_binary.astype(float), (img_w, img_h),
                                  interpolation=cv2.INTER_NEAREST).astype(
        bool)
    mask_and = np.logical_and(map_masks, goal_mask_binary[:, :, None])
    mask_or = np.logical_or(map_masks, goal_mask_binary[:, :, None])
    iou = mask_and.sum(0).sum(0) / mask_or.sum(0).sum(0)
    return iou.argmax(), iou.max()
