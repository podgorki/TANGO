import numpy as np
import cv2
import os
from pathlib import Path
from natsort import natsorted

import habitat_sim
from src.common.utils_sim_traj import find_shortest_path, get_agent_rotation_from_two_positions
from src.common.utils import get_instance_index_to_name_mapping


def get_final_goal_state_reverse(sim, agent_states):
    # find state that is 1 meter away from the starting point
    final_goal_state_idx = None
    for p_idx, p in enumerate(agent_states):
        dis = find_shortest_path(sim, agent_states[0].position, p.position)[0]
        if dis >= 1:
            final_goal_position = p.position
            final_goal_state_idx = p_idx
            break
    if final_goal_state_idx is None:
        raise ValueError('No final goal state found...')
    assert (np.linalg.norm(final_goal_position - agent_states[final_goal_state_idx - 1].position) >= 0.1)
    final_goal_state = habitat_sim.AgentState(
        position=final_goal_position,
        # looking in the reverse direction
        rotation=get_agent_rotation_from_two_positions(final_goal_position,
                                                       agent_states[final_goal_state_idx - 1].position)
    )
    return final_goal_state


def find_reverse_traverse_goal(agent, sim, final_goal_state, map_graph, instance_index_to_name_map=None):
    if instance_index_to_name_map is None:
        instance_index_to_name_map = get_instance_index_to_name_mapping(
            sim.semantic_scene)
    curr_state = agent.get_state()
    agent.set_state(final_goal_state)
    semantic_instance_reverse = sim.get_sensor_observations()['semantic_sensor']
    instance_ids_curr = np.unique(semantic_instance_reverse).flatten()
    cat_names_curr = instance_index_to_name_map[instance_ids_curr][:, 1]
    map_graph_insta_ids = np.array([map_graph.nodes[n]['instance_id'] for n in map_graph.nodes])

    goal_object_cat_names_usual = ['bed', 'chair', 'monitor', 'plant', 'sofa',
                                   'toilet']  # np.unique([str(ep).split('_')[-3] for ep in episodes])
    filter_objects = False

    goal_object_id = None
    for i, insta_id in enumerate(instance_ids_curr):
        if instance_index_to_name_map[insta_id][1] in ['Unknown', 'floor', 'ceiling', 'ceiling lower']:
            continue
        if filter_objects and cat_names_curr[i] not in goal_object_cat_names_usual:
            continue
        if insta_id in map_graph_insta_ids:
            goal_object_id = insta_id
            print(f'Selected goal object for reverse traverse: {cat_names_curr[i]} with id: {goal_object_id}')
            break
    if goal_object_id is None:
        raise ValueError('No goal object (common with map_graph) found in the reverse goal mask...')

    agent.set_state(curr_state)
    return goal_object_id


def get_goal_info_alt_goal(map_path):
    seen_but_unvisited_object = np.load(f"{map_path}/seen_but_unvisited_object.npy", allow_pickle=True)[()]
    goal_instance_id = seen_but_unvisited_object['instance_id']

    goal_img_index = seen_but_unvisited_object['image_id']
    instance_mask = np.load(f"{map_path}/images_sem/{goal_img_index:05d}.npy", allow_pickle=True)

    return goal_img_index, instance_mask, goal_instance_id


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


def get_goal_mask_binary(map_path, task_type):
    if task_type in ["original", "via_alt_goal"]:
        goal_img_index, instance_mask, goal_instance_id = get_goal_info(map_path)

    elif task_type == "alt_goal":
        goal_img_index, instance_mask, goal_instance_id = get_goal_info_alt_goal(map_path)

    else:
        raise ValueError(f"{task_type=} not recognized")

    goal_mask_binary = instance_mask == int(goal_instance_id)

    return goal_img_index, goal_mask_binary, goal_instance_id


def match_map_masks_with_goal_mask(map_masks, goal_mask_binary):
    img_h, img_w, _ = map_masks.shape
    goal_mask_binary = cv2.resize(goal_mask_binary.astype(float), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(
        bool)
    mask_and = np.logical_and(map_masks, goal_mask_binary[:, :, None])
    mask_or = np.logical_or(map_masks, goal_mask_binary[:, :, None])
    iou = mask_and.sum(0).sum(0) / mask_or.sum(0).sum(0)
    return iou.argmax(), iou.max()


def get_goalNodeIdx(map_path, G, nodeID_to_imgRegionIdx, task_type):
    """
    returns the node index of the segment in the map graph that has the highest IoU with the goal object in the last image of the episode's teach run 
    """
    goal_img_index, goal_mask_binary, goal_instance_id = get_goal_mask_binary(map_path, task_type)

    mapNodeInds_in_goalImg = np.argwhere(nodeID_to_imgRegionIdx[:, 0] == goal_img_index).flatten()
    map_masks = utils.nodes2key(mapNodeInds_in_goalImg, 'segmentation', G).transpose(1, 2, 0)

    iou_argmax, _ = match_map_masks_with_goal_mask(map_masks, goal_mask_binary)
    goalNodeIdx = mapNodeInds_in_goalImg[iou_argmax]

    return goalNodeIdx


def get_goalNodeIdx_reverse(map_path, G, goal_instance_id):
    instance_img_paths = natsorted(Path(f"{map_path}/images_sem/").iterdir())
    assert len(instance_img_paths) > 0

    goalImgIdxs = []
    instance_masks = []
    for imgIdx, instance_img_path in enumerate(instance_img_paths):
        instance_img = np.load(instance_img_path, allow_pickle=True)
        if goal_instance_id in np.unique(instance_img):
            goalImgIdxs.append(imgIdx)
            instance_masks.append(instance_img == goal_instance_id)

    nodeID_to_imgRegionIdx = np.array([G.nodes[node]['map'] for node in G.nodes()])
    ious = []
    goalNodeIdxs = []
    for idx, goal_img_index in enumerate(goalImgIdxs):
        mapNodeInds_in_goalImg = np.argwhere(nodeID_to_imgRegionIdx[:, 0] == goal_img_index).flatten()
        map_masks = utils.nodes2key(mapNodeInds_in_goalImg, 'segmentation', G).transpose(1, 2, 0)
        iou_argmax, iou_max = match_map_masks_with_goal_mask(map_masks, instance_masks[idx])
        ious.append(iou_max)
        goalNodeIdxs.append(mapNodeInds_in_goalImg[iou_argmax])

    assert (np.max(ious) != 0), f"goal_instance_id: {goal_instance_id} not found in the map graph"
    goalNodeIdx = goalNodeIdxs[np.argmax(ious)]
    print(f"{goal_instance_id=} found as {goalNodeIdx=}")
    return goalNodeIdx
