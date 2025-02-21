import numpy as np
import math
import magnum as mn
import matplotlib.pyplot as plt
from PIL import Image
import json
from scipy.spatial.transform import Rotation
from spatialmath import SE3, SO3, SE2
from spatialmath.base import trnorm, r2q

import habitat_sim
# from habitat.utils.visualizations import maps
from habitat_sim.utils.common import quat_from_magnum, quat_to_magnum
from utils import display_sample, getK_fromAgent


def maps():
    pass


# Define a function to get a random point near the bounds
def get_random_point_near_bounds(sim, offset=5.0):
    min_bound, max_bound = sim.pathfinder.get_bounds()
    point = np.random.uniform(min_bound, max_bound)
    point = np.clip(point, min_bound + offset, max_bound - offset)
    while not sim.pathfinder.is_navigable(point):
        point = np.random.uniform(min_bound, max_bound)
        point = np.clip(point, min_bound + offset, max_bound - offset)
    return point


def find_shortest_path(sim, p1, p2):
    path = habitat_sim.ShortestPath()
    path.requested_start = p1
    path.requested_end = p2
    found_path = sim.pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance
    path_points = path.points
    # print(f"found_path: {found_path}", geodesic_distance, len(path_points), sim.pathfinder.get_island(p1), sim.pathfinder.get_island(p2))
    # print("found_path : " + str(found_path))
    # print("geodesic_distance : " + str(geodesic_distance))
    # print("path_points : " + str(path_points))
    return geodesic_distance, path_points


def find_max_path_point_near_bounds(sim, source, num_targets=100, offset=5.0):
    max_length, max_point, max_path = 0, None, None

    for _ in range(num_targets):
        target = get_random_point_near_bounds(sim, offset)

        length, path = find_shortest_path(sim, source, target)

        if length > max_length:
            max_length = length
            max_point = target
            max_path = path

    return max_point, max_length, max_path


# Function to find the farthest point from a set of points
def find_farthest_point(sim, points, island_index=0):
    farthest_point = None
    max_min_dist = -np.inf

    for _ in range(100):  # Sample 100 points for checking, increase for more accuracy
        candidate_point = sim.pathfinder.get_random_navigable_point(island_index=island_index)
        min_dist = min([find_shortest_path(sim, candidate_point, p)[0] for p in points])

        if min_dist > max_min_dist:
            max_min_dist = min_dist
            farthest_point = candidate_point

    return farthest_point


# Function to cover the navigable area with farthest point sampling
def cover_navigable_area(sim, initial_point=None, num_points=10, island_index=0):
    if initial_point is None:
        initial_point = sim.pathfinder.get_random_navigable_point(island_index=island_index)

    covered_points = [initial_point]

    while len(covered_points) < num_points:
        next_point = find_farthest_point(sim, covered_points, island_index=island_index)
        if len(find_shortest_path(sim, covered_points[-1], next_point)[1]) == 0: continue
        covered_points.append(next_point)

    return covered_points


def display_point_on_map(sim, nav_point):
    vis_points = [nav_point]
    max_search_radius = 2.0  # @param {type:"number"}
    print(
        "Distance to obstacle: "
        + str(sim.pathfinder.distance_to_closest_obstacle(nav_point, max_search_radius))
    )
    hit_record = sim.pathfinder.closest_obstacle_surface_point(
        nav_point, max_search_radius
    )
    # HitRecord will have infinite distance if no valid point was found:
    if math.isinf(hit_record.hit_dist):
        print("No obstacle found within search radius.")
    else:
        # @markdown Points near the boundary or above the NavMesh can be snapped onto it.
        perturbed_point = hit_record.hit_pos - hit_record.hit_normal * 0.2
        print("Perturbed point : " + str(perturbed_point))
        print(
            "Is point navigable? " + str(sim.pathfinder.is_navigable(perturbed_point))
        )
        snapped_point = sim.pathfinder.snap_point(perturbed_point)
        print("Snapped point : " + str(snapped_point))
        print("Is point navigable? " + str(sim.pathfinder.is_navigable(snapped_point)))
        vis_points.append(snapped_point)

    meters_per_pixel = 0.1

    xy_vis_points = convert_points_to_topdown(
        sim.pathfinder, vis_points, meters_per_pixel
    )
    # use the y coordinate of the sampled nav_point for the map height slice
    top_down_map = maps.get_topdown_map(
        sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    print("\nDisplay the map with key_point overlay:")
    display_map(top_down_map, key_points=xy_vis_points)


def display_long_trajectory(sim):
    # Select a source point close to the bounds
    source_point = get_random_point_near_bounds(sim)

    # Find the target point with the maximum path length, close to the bounds
    max_target_point, max_length, max_path = find_max_path_point_near_bounds(sim, source_point)

    print(f"Source Point: {source_point}")
    print(f"Target Point with Max Path Length: {max_target_point}")
    print(f"Max Path Length: {max_length}")
    print(f"Max Path: {max_path}")

    max_path = interpolate_path_points(max_path, 1)

    display_trajectory(sim, max_path)
    return max_path


def get_trajectory_multipoint_FPS(sim, n=3, island_index=0, display=False):
    covered_points = cover_navigable_area(sim, num_points=n, island_index=island_index)
    # append paths for each point
    paths = []
    for i in range(n - 1):
        length, path = find_shortest_path(sim, covered_points[i], covered_points[i + 1])
        if len(path) == 0: continue
        paths.append(path[:-1])
        print(path)
    paths.append(path[-1:])
    paths = np.concatenate(paths)
    paths = interpolate_path_points(paths, 1)
    agent_states = interpolate_orientation(paths, angle_threshold=np.pi / 6)
    if display: display_trajectory(sim, paths)
    return paths, agent_states


def create_map_trajectory(sim, agent, folderPath, subfolderPath, numPoints=3):
    island_index = sim.pathfinder.num_islands - 1
    path, agent_states = get_trajectory_multipoint_FPS(sim, n=numPoints, island_index=island_index)
    _ = get_images_from_agent_states(sim, agent, agent_states, display=False, saveDir=subfolderPath)
    np.save(f'{folderPath}/path.npy', path)
    np.save(f'{folderPath}/agent_states.npy', agent_states)


# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None, savePath=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    if savePath is not None:
        plt.savefig(savePath)
    plt.show(block=False)


def display_trajectory(sim, path_points, savePath=None, display=False):
    meters_per_pixel = 0.025
    # scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    height = path_points[0][1]  # scene_bb.y().min
    top_down_map = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
    # convert world trajectory points to maps module grid points
    trajectory = [
        maps.to_grid(
            path_point[2],
            path_point[0],
            grid_dimensions,
            pathfinder=sim.pathfinder,
        )
        for path_point in path_points
    ]
    grid_tangent = mn.Vector2(
        trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
    )
    path_initial_tangent = grid_tangent / grid_tangent.length()
    initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
    # draw the agent and trajectory on the map
    maps.draw_path(top_down_map, trajectory)
    maps.draw_agent(
        top_down_map, trajectory[0], initial_angle, agent_radius_px=8
    )
    if display:
        print("\nDisplay the map with agent and path overlay:")
        display_map(top_down_map, savePath=savePath)
    return top_down_map


def display_images_along_path(sim, agent, path_points):
    print("Rendering observations at path points:")
    tangent = path_points[1] - path_points[0]
    agent_state = habitat_sim.AgentState()
    for ix, point in enumerate(path_points):
        if ix < len(path_points) - 1:
            tangent = path_points[ix + 1] - point
            agent_state.position = point
            tangent_orientation_matrix = mn.Matrix4.look_at(point, point + tangent, np.array([0, 1.0, 0]))
            tangent_orientation_q = mn.Quaternion.from_matrix(tangent_orientation_matrix.rotation())
            agent_state.rotation = quat_from_magnum(tangent_orientation_q)
            agent.set_state(agent_state)

            observations = sim.get_sensor_observations()
            rgb = observations["color_sensor"]
            # semantic = observations["semantic_sensor"]
            # depth = observations["depth_sensor"]

            display_sample(rgb)  # semantic, depth)


# Function to calculate angle between two vectors
def angle_between_vectors(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def quat_mn2np(q):
    return np.concatenate([np.array(q.vector), [q.scalar]])


def slerp(q1, q2, fraction):
    # check for angle to take the shorter path along interpolation
    if quat_mn2np(q1).dot(quat_mn2np(q2)) < 0.0:
        q2 = -q2
    return mn.math.slerp(q1, q2, fraction)


def quat_hab_to_Euler(q):
    # expects habitat q, returns euler angles in radians
    return Rotation.from_quat(quat_mn2np(quat_to_magnum(q))).as_euler('xyz')


def quat_hab_from_Euler(e):
    # expects np array [roll,pitch,yaw] in rad, returns np array [x,y,z,w]
    return Rotation.from_euler('xyz', e).as_quat()


def quat_np_to_mn(q):
    return mn.Quaternion([q[:3], q[-1]])


def quat_hab_to_direction(q):
    e = quat_hab_to_Euler(q)
    return np.array([np.cos(e[1]), 0, np.sin(e[1])])


def get_interPoint_orientations(points):
    orientations = []
    for i in range(len(points) - 1):
        ori = mn.Quaternion.from_matrix(mn.Matrix4.look_at(points[i], points[i + 1], mn.Vector3(0, 1.0, 0)).rotation())
        orientations.append(quat_from_magnum(ori))
    return orientations


def interpolate_orientation(path_points, angle_threshold=np.pi / 6, firstPointOrientation=None,
                            lastPointOrientation=None, method='pure'):
    last_tangent = None
    agent_states = []

    for ix, point in enumerate(path_points):
        # handle last point, and its tangent 
        if ix == (len(path_points) - 1):
            if lastPointOrientation is None:
                continue
            else:
                # tangent = quat_hab_to_Euler(lastPointOrientation) - point
                # tangent = -quat_to_magnum(lastPointOrientation).transform_vector(point) + point
                tangent = quat_hab_to_direction(lastPointOrientation)
        else:
            tangent = path_points[ix + 1] - point

        # handle first point, and its tangent
        if ix == 0:
            if firstPointOrientation is None:
                last_tangent = tangent
            else:
                # last_tangent = quat_hab_to_Euler(firstPointOrientation) - point
                # last_tangent = -point + quat_to_magnum(firstPointOrientation).transform_vector(point)
                last_tangent = quat_hab_to_direction(firstPointOrientation)

        if ix == 0:
            initial_orientation = quat_to_magnum(firstPointOrientation)
        else:
            initial_orientation = mn.Quaternion.from_matrix(
                mn.Matrix4.look_at(point - last_tangent, point, mn.Vector3(0, 1.0, 0)).rotation())
            # initial_orientation = quat_np_to_mn(quat_hab_from_Euler(last_tangent))

        if ix == (len(path_points) - 1):
            final_orientation = quat_to_magnum(lastPointOrientation)
        else:
            final_orientation = mn.Quaternion.from_matrix(
                mn.Matrix4.look_at(point, point + tangent, mn.Vector3(0, 1.0, 0)).rotation())
            # final_orientation = quat_np_to_mn(quat_hab_from_Euler(tangent))

        # angle = angle_between_vectors(tangent, last_tangent)
        angle = abs(quat_hab_to_Euler(quat_from_magnum(final_orientation))[1] -
                    quat_hab_to_Euler(quat_from_magnum(initial_orientation))[1])
        if angle < 1e-3:
            agent_states.append(habitat_sim.AgentState(position=point, rotation=quat_from_magnum(initial_orientation)))
            continue
        steps = max(1, int(np.ceil(angle / angle_threshold)))

        for step in range(steps):
            fraction = step / steps
            interpolated_orientation = slerp(initial_orientation, final_orientation, fraction)
            # print(ix,step,fraction,angle,point,initial_orientation,final_orientation,interpolated_orientation)
            agent_states.append(
                habitat_sim.AgentState(position=point, rotation=quat_from_magnum(interpolated_orientation)))
        if method == 'pure':
            agent_states.append(habitat_sim.AgentState(position=point, rotation=quat_from_magnum(final_orientation)))

        last_tangent = tangent

    if method != 'pure':
        agent_states.append(habitat_sim.AgentState(position=point, rotation=quat_from_magnum(final_orientation)))

    return agent_states


def accumulate_diffs(diffs):
    d_cumu = [0]
    for d in diffs:
        d_new = 0 if d == 0 else (d_cumu[-1] + d)
        d_cumu.append(d_new)
    return np.array(d_cumu)[1:]  # remove leading zero


def accumulate_diffs_between_zeros(diffs):
    # expand diffs with leading and trailing zero
    diffs = np.concatenate([[0], diffs, [0]])
    zeroPoints = np.argwhere(diffs == 0).flatten()
    vals = np.array([sum(diffs[zeroPoints[i]: zeroPoints[i + 1]]) for i in range(len(zeroPoints) - 1)])
    # skip last zeroPoint, and add 1 to all zeroPoints to get the indices of the next point
    indices = zeroPoints[:-1] + 1
    diffs_accum = np.zeros_like(diffs)
    diffs_accum[indices] = vals
    return diffs_accum[1:-1]  # remove leading and trailing zero


def getTODO(agent_states):
    pos = np.array([s.position for s in agent_states])
    intraPointDists = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
    angles = np.array([quat_hab_to_Euler(s.rotation)[1] for s in agent_states])
    intraPointAngles = angles[1:] - angles[:-1]
    zeroPoints = np.argwhere(intraPointDists == 0).flatten()
    [sum(intraPointDists[zeroPoints[i]: zeroPoints[i + 1]]) for i in range(len(zeroPoints) - 1)]


def get_checkpoints_ori(agent_states, oris):
    itr = 0
    checkpoints = []
    for i, s in enumerate(agent_states):
        if itr == len(oris): break
        s_angle = quat_hab_to_Euler(s.rotation)[1]
        angle = quat_hab_to_Euler(oris[itr])[1]
        if abs(angle - s_angle) < 1e-3:
            checkpoints.append(i)
            itr += 1
    return checkpoints


def get_checkpoints_trans(agent_states, path):
    itr = 0
    checkpoints = []
    for i, s in enumerate(agent_states):
        if itr == len(path): break
        if sum(abs(s.position - path[itr])) == 0:
            checkpoints.append(i)
            itr += 1
    return checkpoints


def check_agent_states(agent_states):
    # check for pure rotation and pure translation

    pos = np.array([s.position for s in agent_states])
    intraPointDists = np.linalg.norm(pos[1:] - pos[:-1], axis=1)

    angles = np.array([quat_hab_to_Euler(s.rotation)[1] for s in agent_states])
    intraPointAngles = angles[1:] - angles[:-1]

    intersection = np.intersect1d(np.argwhere(intraPointDists == 0).flatten(),
                                  np.argwhere(intraPointAngles == 0).flatten())
    assert (len(intersection) == 0)


def get_tgt_lin(agent_states, path):
    # input path is the non interpolated shortest path points
    check_agent_states(agent_states)

    pos = np.array([s.position for s in agent_states])
    intraPointDists = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
    # traj indices where agent does pure rotation
    pureRotInds = get_checkpoints_trans(agent_states, path)
    tgt_lin = []
    itr = 0
    for i, s in enumerate(agent_states):
        # compute distance to next point in (non-interp) path
        t_lin = np.linalg.norm(s.position - agent_states[pureRotInds[itr]].position)

        # 0 during pure rotation
        if i == len(agent_states) - 1 or intraPointDists[i] == 0:
            t_lin = 0
        if i == pureRotInds[itr] and not itr == len(pureRotInds) - 1:
            itr += 1
        tgt_lin.append(t_lin)
    return tgt_lin


def get_tgt_rot(agent_states, path):
    # input is the interpolated shortest 'path' points, where agent_states is the orientation interpolated version of path
    check_agent_states(agent_states)

    angles = np.array([quat_hab_to_Euler(s.rotation)[1] for s in agent_states])
    intraPointAngles = angles[1:] - angles[:-1]
    # traj indices where agent begins pure translation
    pureTransInds = get_checkpoints_trans(agent_states, path)
    tgt_rot = []
    itr = 1
    for i, s in enumerate(agent_states):
        if i == pureTransInds[itr] and not itr == len(pureTransInds) - 1:
            itr += 1
        # compute angle to next point in (interp) path
        t_rot = quat_hab_to_Euler(agent_states[pureTransInds[itr]].rotation)[1] - quat_hab_to_Euler(s.rotation)[1]

        # 0 during pure rotation
        if i == len(agent_states) - 1 or intraPointAngles[i] == 0:
            t_rot = 0
        tgt_rot.append(t_rot)
    return tgt_rot


def get_tgt_rot_(agent_states, oris):
    # oris obtained from get_interPoint_orientations
    # NOTE: this one has some bug, use the function above
    check_agent_states(agent_states)

    angles = np.array([quat_hab_to_Euler(s.rotation)[1] for s in agent_states])
    intraPointAngles = angles[1:] - angles[:-1]
    # traj indices where agent does pure translation
    pureTransInds = get_checkpoints_ori(agent_states, oris)
    tgt_rot = []
    itr = 0
    for i, s in enumerate(agent_states):
        # compute angle to next point in (non-interp) path
        t_rot = quat_hab_to_Euler(agent_states[pureTransInds[itr]].rotation)[1] - quat_hab_to_Euler(s.rotation)[1]

        # 0 during pure translation
        if i == len(agent_states) - 1 or intraPointAngles[i] == 0:
            t_rot = 0
        if i == pureTransInds[itr] and not itr == len(pureTransInds) - 1:
            itr += 1
        tgt_rot.append(t_rot)
    return tgt_rot


def get_images_from_agent_states(sim, agent, states, display=False, saveDir=None, inc_depth=False, saveDir_depth=None,
                                 inc_sem=False, saveDir_sem=None, collect=True, halfPrecision=False):
    images = []
    for idx, state in enumerate(states):
        agent.set_state(state)
        observations = sim.get_sensor_observations()
        rgb = observations["color_sensor"]
        if collect: images.append(rgb)
        if display: display_sample(rgb)  # Display or process the image
        if saveDir is not None:
            rgb_img = Image.fromarray(rgb, mode="RGBA")
            rgb_img.save(f"{saveDir}/{idx:05d}.png")
        if inc_depth:
            depth = observations["depth_sensor"]
            if halfPrecision:
                depth = depth.astype(np.float16)
            if saveDir_depth is not None:
                np.save(f"{saveDir_depth}/{idx:05d}.npy", depth)
        if inc_sem:
            sem = observations["semantic_sensor"]
            if halfPrecision:
                sem = sem.astype(np.uint16)
            if saveDir_sem is not None:
                np.save(f"{saveDir_sem}/{idx:05d}.npy", sem)
    return images


def interpolate_path_points(path_points, distance_threshold=1.0):
    """
    Interpolate path points so that the distance between consecutive points is 
    approximately equal to the distance_threshold.

    :param path_points: List of path points.
    :param distance_threshold: Maximum distance between interpolated points.
    :return: List of interpolated path points.
    """
    interpolated_points = []

    for i in range(len(path_points) - 1):
        start_point = path_points[i]
        end_point = path_points[i + 1]
        vector = end_point - start_point
        distance = np.linalg.norm(vector)
        num_divisions = max(int(distance / distance_threshold), 1)

        for division in range(num_divisions):
            interpolated_point = start_point + vector * (division / num_divisions)
            interpolated_points.append(interpolated_point)

    interpolated_points.append(path_points[-1])  # Ensure the end point is included

    return interpolated_points


def read_poses_colmap(fname):
    # poses only returned for image names that colmap had (not all images)
    posesDict = {}
    # Read cameras.json
    with open(fname) as f:
        cameras = json.load(f)
        for cam in cameras:
            t = cam['position']
            r = cam['rotation']
            r = np.array(r).reshape(3, 3)
            r = quat_from_magnum(mn.Quaternion.from_matrix(r))
            img_name = cam['img_name']
            posesDict[img_name] = [t, r]
    return posesDict


def hs_quat_to_array(q):
    mn_mat = quat_to_magnum(q).to_matrix()
    return np.array(mn_mat)


def compose_SE3(R, T):
    SE3 = np.eye(4)
    SE3[:3, :3] = R
    SE3[:3, 3] = T
    return SE3


def invert_SE3(SE3):
    SE3_inv = np.eye(4)
    SE3_inv[:3, :3] = SE3[:3, :3].T
    SE3_inv[:3, 3] = -SE3[:3, :3].T @ SE3[:3, 3]
    return SE3_inv


def get_transform_colmap_to_habitat(colmap_cameras_json_filename, habitat_agent_states_filename):
    # Example usage
    # p_c2s = ust.get_transform_colmap_to_habitat("~/workspace/data/inputs/3DGS/output/538144b1-6/cameras.json","~/workspace/s/sg_habitat/out/maps/multiPointTrajs/not2use_skokloster-castle_20240113171224854916/agent_states.npy")
    posesDict = read_poses_colmap(colmap_cameras_json_filename)
    pathPoses = np.load(habitat_agent_states_filename, allow_pickle=True)

    # inds = [int(k) for k in posesDict.keys()]
    # pathPoses = pathPoses[inds]

    poses_colmap, poses_habitat = [], []
    for k in posesDict.keys():
        r_ci = hs_quat_to_array(posesDict[k][1])
        p_ci = compose_SE3(r_ci, posesDict[k][0])
        poses_colmap.append(p_ci)

        r_si = hs_quat_to_array(pathPoses[int(k)].rotation)
        p_si = compose_SE3(r_si, pathPoses[int(k)].position)
        poses_habitat.append(p_si)
    poses_colmap = np.array(poses_colmap)
    poses_habitat = np.array(poses_habitat)
    # np.save("poses_colmap.npy",poses_colmap)
    # np.save("poses_habitat.npy",poses_habitat)

    dists_3D_colmap = np.linalg.norm(poses_colmap[:, None, :3, 3] - poses_colmap[None, :, :3, 3], axis=-1)
    dists_3D_habitat = np.linalg.norm(poses_habitat[:, None, :3, 3] - poses_habitat[None, :, :3, 3], axis=-1)
    dists_ratio = dists_3D_habitat / dists_3D_colmap
    inds = np.triu_indices_from(dists_ratio, 1)
    # plt.imshow(dists_ratio)
    # plt.colorbar()
    # plt.show()
    dists_ratio = dists_ratio[inds]
    scale = np.median(dists_ratio)
    # print(np.mean(dists_ratio), np.std(dists_ratio), np.max(dists_ratio), np.min(dists_ratio), np.median(dists_ratio)
    poses_colmap[:, :3, 3] *= scale

    # plot 3D points for both colmap and habitat
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(poses_colmap[:,0,3],poses_colmap[:,1,3],poses_colmap[:,2,3],label='colmap')
    # ax.scatter(poses_habitat[:,0,3],poses_habitat[:,1,3],poses_habitat[:,2,3],label='habitat')
    # ax.legend()
    # plt.show()

    tfms = []
    for p_ci, p_si in zip(poses_colmap, poses_habitat):
        p_cs = p_ci @ invert_SE3(p_si)
        print(p_cs)
        tfms.append(p_cs)

    errs_t, errs_r = [], []
    for p_cs in tfms:
        err_t, err_r = 0, 0
        for p_ci, p_si in zip(poses_colmap, poses_habitat):
            eMat = (p_cs @ p_si) @ invert_SE3(p_ci)
            e_trans = np.linalg.norm(eMat[:3, 3])
            e_rot = np.trace(eMat[:3, :3])
            err_t += e_trans
            err_r += e_rot
        errs_r.append(err_r / len(poses_colmap))
        errs_t.append(err_t / len(poses_colmap))
    idx_min_err_t = np.argmin(errs_t)
    idx_min_err_r = np.argmin(errs_r)
    print(f"idx_min_err_t: {idx_min_err_t}, idx_min_err_r: {idx_min_err_r}")
    print(f"Idx {idx_min_err_t}, t err: {errs_t[idx_min_err_t]}, r err: {errs_r[idx_min_err_t]}")
    print(f"Idx {idx_min_err_r}, t err: {errs_t[idx_min_err_r]}, r err: {errs_r[idx_min_err_r]}")
    tfm = tfms[np.argmin(errs_r)]
    print(tfms[np.argmin(errs_r)])
    print(tfms[np.argmin(errs_t)])

    # transform habitat poses to colmap poses
    poses_habitat_tfm = []
    for p_si in poses_habitat:
        p_ci = tfm @ p_si
        poses_habitat_tfm.append(p_ci)
    poses_habitat_tfm = np.array(poses_habitat_tfm)

    # plot 2D points for both colmap and habitat
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(poses_habitat[:, 0, 3], poses_habitat[:, 2, 3], '-o', label='habitat', alpha=0.5)
    ax.plot(poses_colmap[:, 0, 3], poses_colmap[:, 2, 3], '-o', label='colmap', alpha=0.5)
    ax.plot(poses_habitat_tfm[:, 0, 3], poses_habitat_tfm[:, 2, 3] - 20, '-o', label='habitat_to_colmap (y-20)',
            alpha=0.5)
    # plot lines between corresponding points
    # for p_ci,p_si in zip(poses_colmap,poses_habitat):
    # ax.plot([p_ci[0,3],p_si[0,3]],[p_ci[2,3],p_si[2,3]],'k--', alpha=0.25)
    # legend outside
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    return tfm


def SE3_from4x4(pose):
    # check is False, do see https://github.com/bdaiinstitute/spatialmath-python/issues/28
    # return SE3.Rt(pose[:3,:3], pose[:3,-1],check=False)
    # if pose is instance list of R (3x3),t(3), conver tto pose
    if isinstance(pose, list) and len(pose) == 2:
        pose4x4 = np.eye(4)
        R, t = pose
        pose4x4[:3, :3] = R
        pose4x4[:3, -1] = t.flatten()
        pose = pose4x4
    return SE3(trnorm(np.array(pose)))


def plotAgentStates_SE3(states, tfm=None, wrtFirst=False, toSE2=False, areSE3=False):
    # if in SE3 format
    if areSE3:
        poses = states
    # else convert habitat states into SE3
    else:
        Rs = [hs_quat_to_array(s.rotation) for s in states]
        ts = np.array([s.position for s in states])
        poses = [SE3_from4x4([Rs[i], ts[i]]) for i in range(len(Rs))]

    # mainly to convert base frame to camera frame
    if tfm is not None:
        poses = [p @ SE3(tfm) for p in poses]

    # poses wrt the first pose
    # TODO: wrtFirst should be after toSE2 but ValueError: argument is not valid SE(2) matrix
    if wrtFirst:
        poses = [poses[0].inv() @ p for p in poses]

    # convert SE3 to SE2 (in habitat's coordinate system)
    if toSE2:  # assuming habitat's z-x motion
        angles = [np.arctan2(p.R[0, 2], p.R[2, 2]) for p in poses]
        poses = [SE2(p.t[2], p.t[0], angles[i]) for i, p in enumerate(poses)]
    # [print(i,p) for i,p in enumerate(poses)]
    [p.plot(frame=f"{i}", length=0.5, width=0.1, wtl=0.05, arrow=False) for i, p in enumerate(poses)]
    plt.axis('equal')
    plt.show()
    return poses


def unproject2D(r, c, z, K, appendOnes=False, retMask=True):
    mask = z != 0
    x = z * (c - K[0, 2]) / K[0, 0]
    y = z * (r - K[1, 2]) / K[1, 1]
    p3d = np.column_stack((x, y, z))
    if appendOnes:
        p3d = np.column_stack((p3d, np.ones_like(z)))
    if retMask:
        return p3d, mask
    else:
        return p3d


def get_point_camera2world(currState, p_c):
    T_BC = SE3.Rx(np.pi).A  # camera to base
    T_WB = SE3_from4x4([hs_quat_to_array(currState.rotation), currState.position]).A  # base to world
    p_w = T_WB @ T_BC @ p_c
    return p_w.T


def get_points_world_from_depth(depth, agent):
    H, W = depth.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))

    K = getK_fromAgent(agent)
    p3d, mask = unproject2D(ys.flatten(), xs.flatten(), depth.flatten(), K, appendOnes=True, retMask=True)

    p3d_w = get_point_camera2world(agent.get_state(), p3d.T)
    return p3d_w


def get_pathlength_GT(sim, agent, depth, semantic, goalPosition_w, randColors=None, display=False,
                      instaIdx2catName=None):
    H, W = depth.shape
    K = getK_fromAgent(agent)
    instances = np.unique(semantic)
    numSamples = 20
    areaThresh = int(np.ceil(0.001 * H * W))
    curr_state = agent.get_state()

    # gather subset of points per instance
    p2d_c = []
    inds, pointsAll = [], []
    for i, insta_idx in enumerate(instances):
        points = np.argwhere(semantic == insta_idx)
        subInds = np.linspace(0, len(points) - 1, numSamples).astype(int)
        p2d_c.append(points[subInds])
        inds.append(i * np.ones(len(subInds)))
        pointsAll.append(points)
    inds = np.concatenate(inds)
    p2d_c = np.concatenate(p2d_c, 0)

    # get 3D points in world frame
    p3d_c = unproject2D(p2d_c[:, 0], p2d_c[:, 1], depth[p2d_c[:, 0], p2d_c[:, 1]], K, appendOnes=True, retMask=False)
    p3d_w = get_point_camera2world(agent.get_state(), p3d_c.T)

    # get shortest path from navigable points to goal
    # check if navigable point is farther than the original point
    # TODO: instead use normal to image plane as a measure
    p_w_nav = np.array([sim.pathfinder.snap_point(p) for p in p3d_w[:, :3]])
    pls = np.array([find_shortest_path(sim, p, goalPosition_w)[0] for p in p_w_nav])
    eucDists_agent_to_p3dw = np.linalg.norm(curr_state.position - p3d_w[:, :3], axis=1)
    eucDists_agent_to_pwnav = np.linalg.norm(curr_state.position - p_w_nav[:, :3], axis=1)
    distsMask = eucDists_agent_to_p3dw > eucDists_agent_to_pwnav
    # reduce over multiple points in the same instance
    plsImg = np.zeros([H, W])
    pl_min_insta, points_min_pl, points_min_pl_nav = [], [], []
    for i in range(len(instances)):
        subInds = inds == i
        pls_insta = pls[subInds]
        distsMask_insta = distsMask[subInds]
        if distsMask_insta.sum() == 0:
            pl_min = np.inf
        else:
            pl_min = np.min(pls_insta[distsMask_insta])
        if pl_min == np.inf:
            if instaIdx2catName is not None:
                print(f"Instance '{instaIdx2catName[instances[i], 1]}' has no path to goal")
            pl_min = 99
        if len(pointsAll[i]) <= areaThresh:
            if instaIdx2catName is not None:
                print(
                    f"Instance '{instaIdx2catName[instances[i], 1]}' has {len(pointsAll[i])} points < {areaThresh} threshold.")
            pl_min = 99
        if instances[i] == 0:
            if instaIdx2catName is not None:
                print(f"Removing instance '{instaIdx2catName[instances[i], 1]}'")
            pl_min = 99
        pl_min_insta.append(pl_min)
        plsImg[pointsAll[i][:, 0], pointsAll[i][:, 1]] = pl_min

        points_min_pl.append(p3d_w[subInds][np.argmin(pls_insta)])
        points_min_pl_nav.append(p_w_nav[subInds][np.argmin(pls_insta)])
    pls = np.array(pl_min_insta)
    points_min_pl = np.array(points_min_pl)
    points_min_pl_nav = np.array(points_min_pl_nav)
    plsDict = {instances[i]: pls[i] for i in range(len(instances))}
    plsImg = plsImg.reshape([H, W])

    if display:
        if randColors is None:
            colors = semantic.flatten()
        else:
            colors = randColors[semantic].reshape([-1, 3]) / 255.0
        plt.scatter(p3d_w[:, 0], p3d_w[:, 2], c=colors)
        plt.scatter(points_min_pl[:, 0], points_min_pl[:, 2], c='w', marker='*')
        plt.scatter(points_min_pl_nav[:, 0], points_min_pl_nav[:, 2], c=pls)
        # plt.scatter(p_w_nav[:,0], p_w_nav[:,2],c=colors[::1000])
        plt.colorbar()
        plt.show()
    return pls, plsDict, plsImg
