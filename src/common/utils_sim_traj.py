import numpy as np
from spatialmath import SE3
from spatialmath.base import trnorm

import habitat_sim
from habitat_sim.utils.common import quat_to_magnum
from src.common.utils import getK_fromAgent


def find_shortest_path(sim, p1, p2):
    path = habitat_sim.ShortestPath()
    path.requested_start = p1
    path.requested_end = p2
    _ = sim.pathfinder.find_path(path)  # must be called to get path
    geodesic_distance = path.geodesic_distance
    path_points = path.points
    return geodesic_distance, path_points


def hs_quat_to_array(q):
    mn_mat = quat_to_magnum(q).to_matrix()
    return np.array(mn_mat)


def SE3_from4x4(pose):
    # check is False, do see https://github.com/bdaiinstitute/spatialmath-python/issues/28
    if isinstance(pose, list) and len(pose) == 2:
        pose4x4 = np.eye(4)
        R, t = pose
        pose4x4[:3, :3] = R
        pose4x4[:3, -1] = t.flatten()
        pose = pose4x4
    return SE3(trnorm(np.array(pose)))


def unproject2D(r, c, z, K):
    x = z * (c - K[0, 2]) / K[0, 0]
    y = z * (r - K[1, 2]) / K[1, 1]
    p3d = np.column_stack((x, y, z))
    p3d = np.column_stack((p3d, np.ones_like(z)))
    return p3d


def get_point_camera_2_world(currState, p_c):
    T_BC = SE3.Rx(np.pi).A  # camera to base
    T_WB = SE3_from4x4([hs_quat_to_array(currState.rotation), currState.position]).A  # base to world
    p_w = T_WB @ T_BC @ p_c
    return p_w.T


def get_navigable_points_on_instances(sim, K, curr_state, depth, semantic, num_samples: int = 20):
    instances_valid = np.unique(semantic)

    # gather subset of points per instance
    p2d_c = []
    indexes, points_all = [], []
    for i, instance_index in enumerate(instances_valid):
        points = np.argwhere(semantic == instance_index)
        sub_indexes = np.linspace(0, len(points) - 1, num_samples).astype(int)
        p2d_c.append(points[sub_indexes])
        indexes.append(i * np.ones(len(sub_indexes)))
        points_all.append(points)
    indexes = np.concatenate(indexes)
    p2d_c = np.concatenate(p2d_c, 0)

    # get 3D points in world frame
    p3d_c = unproject2D(p2d_c[:, 0], p2d_c[:, 1], depth[p2d_c[:, 0], p2d_c[:, 1]], K)
    p3d_w = get_point_camera_2_world(curr_state, p3d_c.T)
    p_w_nav = np.array([sim.pathfinder.snap_point(p) for p in p3d_w[:, :3]])

    # WIP: compute path lengths from segment to segment (intra image)
    return p3d_c, p3d_w, p_w_nav, instances_valid, indexes, points_all


def get_goal_mask(sim, agent, depth, semantic, final_goal_position):
    H, W = depth.shape
    area_threshold = int(np.ceil(0.001 * H * W))
    curr_state = agent.get_state()
    K = getK_fromAgent(agent)

    points_data = get_navigable_points_on_instances(sim, K, curr_state, depth, semantic)
    if points_data is None:
        return None, None, None
    p3d_c, p3d_w, p_w_nav, instances, indexes, points_all = points_data

    # get shortest path from navigable points to goal
    # check if navigable point is farther than the original point
    pls = np.array([find_shortest_path(sim, p, final_goal_position)[0] for p in p_w_nav])
    eucDists_agent_to_p3dw = np.linalg.norm(curr_state.position - p3d_w[:, :3], axis=1)
    eucDists_agent_to_pwnav = np.linalg.norm(curr_state.position - p_w_nav[:, :3], axis=1)
    distance_masks = eucDists_agent_to_p3dw > eucDists_agent_to_pwnav

    # reduce over multiple points in the same instance
    pls_image = np.zeros([H, W])
    for i in range(len(instances)):
        sub_indexes = indexes == i
        pls_instance = pls[sub_indexes]
        distance_masks_instance = distance_masks[sub_indexes]
        if distance_masks_instance.sum() == 0:
            pl_min = np.inf
        else:
            pl_min = np.min(pls_instance[distance_masks_instance])
        if pl_min == np.inf or len(points_all[i]) <= area_threshold or instances[i] == 0:
            pl_min = 99
        pls_image[points_all[i][:, 0], points_all[i][:, 1]] = pl_min
    pls_image = pls_image.reshape([H, W])
    return pls_image
