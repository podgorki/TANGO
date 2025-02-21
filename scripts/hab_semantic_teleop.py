# %%
import utils
import pickle
import nav_parser
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional
import os
from pathlib import Path
import sys
import habitat_sim
from natsort import natsorted
from libs.path_finding.path_finder import AStar, Dijkstra
from libs.path_finding.graphs import CostMapGraphNX
import libs.path_finding.plot_utils as plot_utils
import kornia as K
import torch
import quaternion as qt
from libs.control.pid import SteerPID
from libs.control.utils import apply_velocity, log_control
import logging
import datetime
import networkx as nx
import time
from scipy.interpolate import splrep, BSpline
from utils_sim_traj import get_pathlength_GT

logger = logging.getLogger(__name__)


def getSimAgent(test_scene, **kwargs):
    sim_settings = utils.get_sim_settings(scene=test_scene,
                                          width=kwargs.get('width', 640), height=kwargs.get('height', 480),
                                          hfov=kwargs.get('hfov', 58), sensor_height=1.31)
    cfg = utils.make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(sim_settings["default_agent"])
    return agent, sim


def compute_point_tangents(points: np.ndarray, ned: bool = True):
    point_next = np.roll(points, axis=0, shift=-1)
    point_diff = point_next - points
    xs = point_diff[:, 0]
    zs = point_diff[:, 1]
    thetas = np.arctan2(xs, zs)  # estimate tangents with points in front
    thetas[-1] = thetas[-2]  # estimate tangent from previous point
    thetas = np.roll(thetas, axis=0, shift=1)  # make sure we aim at the next point
    thetas[-1] = thetas[0]  # we dont know which way to face because we have no next point
    thetas[0] = 0  # initially facing forward
    return thetas[..., None]


def compute_pose_estimate(thetaj_current: float,
                          velocity: float,
                          xj: float,
                          yj: float) -> tuple:
    xj = xj + velocity * np.sin(thetaj_current)
    yj = yj + velocity * np.cos(thetaj_current)
    return xj, yj, thetaj_current


def compute_distance_field(traversable, kernel_size=3, h=.5, device='cpu'):
    # Taking a matrix of size k as the kernel
    dist_0 = traversable[None, None, ...]  # traversable
    dist_1 = (1 - traversable)[None, None, ...]  # non-traversable
    dist_fields = K.contrib.distance_transform(torch.concatenate((dist_0, dist_1), 0),
                                               kernel_size=kernel_size, h=h)
    dist_field = 10 * dist_fields[0] + -1 * dist_fields[1]
    dist_field += -1 * dist_field.min()  # make all values positive or the path finders have a heart attack
    return dist_field


def unproject_points(us: torch.Tensor, vs: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor):
    homogeneous_pts = torch.concatenate((
        us[..., None],
        vs[..., None],
        torch.ones(size=(depth.shape[0], 1), device=us.device)
    ), dim=1).T.to(float)
    unprojected_points = (torch.matmul(torch.linalg.inv(intrinsics), homogeneous_pts)).T
    unprojected_points *= depth
    return unprojected_points


def compute_relative_bev(
        traversable: torch.Tensor,
        depth: torch.Tensor,
        image_width: int,
        image_height: int,
        robot_height: float,
        intrinsics: torch.Tensor,
        goalies: torch.Tensor,
        grid_size: float = 0.5,
):
    u = torch.arange(0, image_width, requires_grad=False, device=depth.device)
    v = torch.arange(0, image_height, requires_grad=False, device=depth.device)
    vs, us = torch.meshgrid(v, u)
    us = us.reshape(-1)
    vs = vs.reshape(-1)
    unprojected_points = unproject_points(us, vs, depth, intrinsics)

    # prep for voxelizing
    start = time.time()
    unprojected_points -= unprojected_points % grid_size
    unprojected_points = unprojected_points.round(decimals=3)

    # set up occupancy map (x, y, z) - this is not the sim coordinate system
    grid_min = torch.tensor([-5, -5, 0])
    grid_max = torch.tensor([5, 5, 10])
    cells = ((grid_max - grid_min) / grid_size).to(int)
    occupancy = torch.zeros(cells[2].item(), cells[0].item(), device=depth.device, requires_grad=False,
                            dtype=torch.float16)
    occupancy_goal = torch.ones_like(occupancy, device=depth.device, requires_grad=False, dtype=int) * 99
    x_range = torch.arange(grid_min[0].item(), grid_max[0].item(), grid_size).cuda().round(decimals=3).half()
    z_range = torch.arange(grid_min[2].item(), grid_max[2].item(), grid_size).cuda().round(decimals=3).half()

    # voxelize and fill occupancy
    unprojected_points = unprojected_points.half()
    while unprojected_points.shape[0] > 0:
        mask = torch.all(torch.eq(unprojected_points, unprojected_points[0]), dim=1)
        # fill occupancy at that voxel
        in_min = (unprojected_points[mask][0, :] > grid_min.cuda()).all()
        in_max = (unprojected_points[mask][0, :] < grid_max.cuda()).all()
        # ensure the points are in range
        if (torch.logical_and(in_min, in_max) == True).cpu().item():
            x, _, z = unprojected_points[mask][0, :]
            x = torch.where(x_range == x.item())[0].item()
            z = torch.where(z_range == z.item())[0].item()
            occupancy[z, x] = (traversable[mask] == True).any().item()
            occupancy_goal[z, x] = goalies[mask].min()
            # update points
        not_mask = torch.logical_not(mask)
        unprojected_points = unprojected_points[not_mask]
        goalies = goalies[not_mask]
        traversable = traversable[not_mask]

    stop = time.time()
    print(f'voxelize time: {(stop - start):.6f}')
    # h, w = occupancy.shape
    # occupancy[:3, (w // 2) - 2:(w // 2) + 2] = 1  # this is a hack
    return occupancy.float(), grid_size, x_range, z_range, occupancy_goal


def split_observations(observations, sem_insta_2_cat_map):
    rgb_obs = observations["color_sensor"]
    depth = observations["depth_sensor"]
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    display_img = np.array(rgb_img.convert('RGB'))
    semantic = observations["semantic_sensor"]  # an array of instance ids
    semantic = sem_insta_2_cat_map[semantic][:, :, 1]
    return display_img, depth, semantic


def get_traversibility(semantic, traversible_classes):
    traversible = np.zeros_like(semantic, dtype=float)
    for trav_class in traversible_classes:
        traversible += (semantic == trav_class).astype(float)

    # make the border closer to the platform non-traversivble for the cost map
    kernel_erode = np.ones((7, 7), np.uint8)
    traversible_safe = cv2.erode(traversible, kernel_erode, iterations=1)
    return traversible, traversible_safe


def find_boundaries(bev_relative, x_bev_range, z_bev_range):
    bev_boundaries = K.filters.canny(bev_relative[None, None, ...], low_threshold=.5, high_threshold=.99)[1].squeeze(0,
                                                                                                                     1)
    vs_boundary, us_boundary = torch.meshgrid(torch.arange(bev_boundaries.shape[0], device=bev_relative.device),
                                              torch.arange(bev_boundaries.shape[1], device=bev_relative.device))
    us_boundary = x_bev_range[us_boundary[bev_boundaries > 1]]
    vs_boundary = z_bev_range[vs_boundary[bev_boundaries > 1]]
    bev_boundary_coords = torch.concatenate((us_boundary[..., None], vs_boundary[..., None]), dim=1)
    return bev_boundary_coords


def setup_sim():
    Path('logs').mkdir(exist_ok=True, parents=True)
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    logging.basicConfig(
        filename=f"logs/{datetime_string}_run.log",
        encoding="utf-8",
        filemode="a",
        format="%(levelname)s - %(message)s",
        level=logging.INFO,
    )

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    sys.path.append(parent_dir)

    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    # get args
    args = nav_parser.parse_args("-ds hm3d -v 0.05 -c 'ceiling' -d".split(" "))
    seed = args.seed
    max_steps = 1000  # args.max_steps
    np.random.seed(seed)

    update_nav_mesh = False
    glbDir = "./data/hm3d_v0.2/train/"
    # test_scene_name = "5cdEh9F2hJL"  # for val
    test_scene_name = "1S7LAXRdDqK"  # for train
    glbDirs = natsorted(os.listdir(glbDir))
    glbDir_test_scene = [d for d in glbDirs if test_scene_name in d][0]
    test_scene = f"{glbDir}/{glbDir_test_scene}/{test_scene_name}.basis.glb"
    logger.info(f"Dataset: {args.dataset}, "
                f"Running with seed {seed} "
                f"and max steps {max_steps}, "
                f"test scene: {test_scene}")

    # create and configure a new VelocityControl structure
    vel_control = habitat_sim.physics.VelocityControl()
    vel_control.controlling_lin_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.controlling_ang_vel = True
    vel_control.ang_vel_is_local = True
    # agent, sim = getSimAgent(test_scene, **{"width": 256, "height": 256, "hfov": 120})
    sim, agent, action_names, cfg = utils.get_sim_agent(test_scene, update_nav_mesh)
    agent_state = agent.get_state()
    sim.agents[0].agent_config.sensor_specifications[1].normalize_depth = True

    # place the robot in a 'good' location
    # agent_state.position = agent_state.position
    # agent_state.rotation = qt.quaternion(-0.0871556401252747, 0, 0.99619472026825, 0)
    # agent.set_state(agent_state)
    logger.info(f'agent start position: {agent_state.position}, rotation: {agent_state.rotation}')
    return sim, agent, vel_control, max_steps


def setup_sim_plots():
    fig, ax = plt.subplots(2, 4)
    plt.ion()
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    return fig, ax


def build_intrinsics(image_width: int,
                     image_height: int,
                     field_of_view_radians_u: float,
                     field_of_view_radians_v: Optional[float] = None,
                     device='cpu') -> torch.Tensor:
    if field_of_view_radians_v is None:
        field_of_view_radians_v = field_of_view_radians_u
    center_u = image_width / 2
    center_v = image_height / 2
    fov_u = (image_width / 2.) / np.tan(field_of_view_radians_u / 2.)
    fov_v = (image_height / 2.) / np.tan(field_of_view_radians_v / 2.)
    intrinsics = np.array([
        [fov_u, 0., center_u],
        [0., fov_v, center_v],
        [0., 0., 1]
    ])
    intrinsics = torch.from_numpy(intrinsics).to(device)
    return intrinsics


def change_edge_attr(G):
    for e in G.edges(data=True):
        if 'margin' in e[2]:
            e[2]['margin'] = 0.0
    return G


def getGoalMask(G4, sem, goal_object_id=None):
    G4 = change_edge_attr(G4)
    G_insta_ids = np.array([G4.nodes[n]['instance_id'] for n in G4.nodes])
    if goal_object_id is None:
        goalNodeIdx = len(G4.nodes) - 1
    else:
        goalNodeIdx = np.argwhere(G_insta_ids == 42).flatten()[-1]
    plsDict = nx.single_source_dijkstra_path_length(G4, goalNodeIdx, weight='margin')
    pls = np.array([plsDict[n] for n in range(len(G4.nodes))])

    matches = sem[:, :, None] == G_insta_ids[None, None, :]
    matchedNodeInds = np.argmax(matches, 2)
    matchedNodeInds_pls = pls[matchedNodeInds]

    semCounts = np.bincount(sem.flatten())
    semInvalid = semCounts[sem] < 10  # small segments
    matchedNodeInds_pls[semInvalid] = 100  # assign high pl
    matchesInvalid = np.sum(matches, 2) == 0  # no match
    matchedNodeInds_pls[matchesInvalid] = 101  # assign high pl
    return matchedNodeInds_pls


def go_robot():
    sim, agent, vel_control, max_steps = setup_sim()
    sem_insta_2_cat_map = utils.get_instance_to_category_mapping(sim.semantic_scene)

    # for debugging on my system ------
    import matplotlib
    matplotlib.use('Qt5Agg')
    fig, ax = setup_sim_plots()

    # config params
    robot_height = agent.agent_config.height

    traversable_classes = [25, 30, 37]
    time_delta = 0.1
    distance_slop = 0.05
    velocity_control = 0.05

    # setup controller
    pid_steer_values = [.5, 0, 0]  # [0.035, 0.1, 0.0000001]
    pid_steer = SteerPID(Kp=pid_steer_values[0], Ki=pid_steer_values[1], Kd=pid_steer_values[2])
    logger.info(f'steering PID values: Kp={pid_steer_values[0]}, Ki={pid_steer_values[1]}, Kd={pid_steer_values[2]}')

    image_height, image_width = sim.config.agents[0].sensor_specifications[0].resolution
    # hfov_radians = float(sim.agents[0].agent_config.sensor_specifications[1].hfov) * np.pi / 180.
    hfov_radians = np.pi * 120 / 180
    device = torch.cuda.current_device()
    intrinsics = build_intrinsics(image_width, image_height, hfov_radians, device=device)
    enable_logging = False

    # load robohop graph
    agent_states = np.load(
        '/storage/Datasets/AIhabitat/robohop/navigation/hm3d_iin_train/1S7LAXRdDqK_0000000_plant_42_/agent_states.npy',
        allow_pickle=True)
    agent.set_state(agent_states[0])
    # map_graph = pickle.load(open(f"/home/stefan/work/collaborations/sg_habitat/data/nodes_graphObject_4.pickle", 'rb'))  # not right now
    # todo: currently the control cant handle when there is no traversable path it seems to hang (maybe astar is taking ages)
    # Main loop

    # todo: for evaluation
    # curr_state = agent.get_state()  # world coords
    # path_length = find_shortest_path(sim, curr_state.position, goal_position)[0]  # give absolute path goal

    while True:
        observations = sim.get_sensor_observations()
        display_img, depth, semantic = split_observations(
            observations,
            sem_insta_2_cat_map,
        )
        traversable_safe, _ = get_traversibility(semantic, traversable_classes)
        h, w = traversable_safe.shape
        # goal_mask = getGoalMask(map_graph, semantic)
        goal_positon = agent_states[-1].position
        randColors = np.random.randint(0, 255, (1000, 3))
        _, _, goal_mask = get_pathlength_GT(sim, agent, depth, semantic, goal_positon, randColors)
        bev_relative_safe, grid_size, x_bev_range, z_bev_range, occupancy_goal_safe = compute_relative_bev(
            traversable=torch.from_numpy(traversable_safe.reshape(-1)).half().to(device),
            depth=torch.from_numpy(depth.reshape(-1)[:, None]).repeat(1, 3).to(device),
            image_width=w,
            image_height=h,
            robot_height=robot_height,
            intrinsics=intrinsics,
            goalies=torch.from_numpy(goal_mask.reshape(-1)).int().to(device),
            grid_size=0.5
        )
        dist_field_relative_bev = 1 - bev_relative_safe[None, ...]

        # for plotting colors
        # goal_mask[goal_mask == 0] = 20
        # goal_mask[goal_mask == 101] = 20
        # goal_mask[goal_mask == 100] = 20
        #
        # for plotting colours but also to help finding the min goal
        # occupancy_goal_safe[occupancy_goal_safe == 0] = 20  # final goal should be a zero (for goalyness, gt based is distance so should just be small)
        # occupancy_goal_safe[occupancy_goal_safe == 101] = 20
        # occupancy_goal_safe[occupancy_goal_safe == 100] = 20

        # for gt I pick the smallest value in the goal map
        min_y, min_x = torch.where(occupancy_goal_safe == occupancy_goal_safe.min())
        # todo: need to pick the min with gt - seems not to have a unique min... speak to Sourav
        dist_field_relative_bev = K.filters.gaussian_blur2d(
            dist_field_relative_bev[None, ...], (3, 3), (.5, .5)
        ).squeeze(0, 1)
        # map ids to colors
        h_bev, w_bev = dist_field_relative_bev.shape

        # Find control path with dijkstra
        start_bev = (w_bev // 2, 0)
        goal_idx = torch.sqrt((min_x - start_bev[0]) ** 2 + (min_y - start_bev[1]) ** 2).max(0)[1]
        # get the furthest goal point
        goal_y = min_y[goal_idx].item()
        goal_x = min_x[goal_idx].item()
        goal_bev = (goal_x, goal_y)
        cmg = CostMapGraphNX(
            width=w_bev,
            height=h_bev,
            cost_map=dist_field_relative_bev
        )
        path_traversable_bev = cmg.get_path(start_bev, goal_bev)
        # move the robot
        if path_traversable_bev.shape[0] > 0:
            skips = 2
            traversible_bev_xs = x_bev_range.cpu()[path_traversable_bev[:, 0]]
            traversible_bev_zs = z_bev_range.cpu()[path_traversable_bev[:, 1]]
            if path_traversable_bev.shape[0] > skips:
                traversible_bev_xs = traversible_bev_xs[::skips]
                traversible_bev_zs = traversible_bev_zs[::skips]

            try:
                t = np.concatenate(
                    (np.array([0]),
                     np.cumsum(np.diff(traversible_bev_xs, 1) ** 2 + np.diff(traversible_bev_zs, 1) ** 2))
                ) / traversible_bev_xs.shape[0]
                ti = np.linspace(0, t[-1], 20)
                tck_x = splrep(t, traversible_bev_xs, s=0)
                tck_z = splrep(t, traversible_bev_zs, s=0)
                traversible_bev_xs = BSpline(*tck_x)(ti)
                traversible_bev_zs = BSpline(*tck_z)(ti)
            except TypeError:
                pass  # sometimes things just dont go to plan so default to janky paths

            traversible_bev = np.concatenate((traversible_bev_xs[:, None], traversible_bev_zs[:, None]), axis=1)
            thetas = compute_point_tangents(traversible_bev)
            point_poses = np.concatenate((traversible_bev, thetas), axis=1)
            plot_utils.plot_sensors(
                ax=ax,
                display_img=display_img,
                semantic=semantic,
                depth=depth,
                relative_bev=K.tensor_to_image(bev_relative_safe),
                occupancy_goal=K.tensor_to_image(occupancy_goal_safe),
                goal_mask=goal_mask,
            )
            ax[1, 3].scatter(min_x.cpu().numpy(), min_y.cpu().numpy(), marker='x', color='black')
            ax[1, 3].scatter(min_x.cpu().numpy()[goal_idx], min_y.cpu().numpy()[goal_idx], marker='o', color='yellow')
            ax[1, 3].scatter(start_bev[0], start_bev[1], marker='o', color='red')
            # set up thresholds
            plot_utils.plot_path_points(
                ax=[ax[1, 2], ax[1, 0]],
                points=point_poses,
                dist_field_relative_bev=K.tensor_to_image(dist_field_relative_bev),
                colour='red'
            )
            # setting intermediate starting point
            xj = point_poses[0, 0]
            yj = point_poses[0, 1]
            thetaj = point_poses[0, 2]
            thetaj_current = thetaj  # initialize facing the correct way
            for i, (xi, yi, thetai) in enumerate(point_poses[1:2, :]):
                distance_error = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                theta_error = thetai - thetaj
                while distance_error > distance_slop:
                    # observations = sim.get_sensor_observations()
                    # display_img, depth, semantic = split_observations(
                    #     observations,
                    #     sem_insta_2_cat_map,
                    # )
                    # goalMask = getGoalMask(mapGraph, semantic)
                    plot_utils.plot_position(axs=[ax[1, 2], ax[1, 0]],
                                             x_image=path_traversable_bev[i, 0],
                                             y_image=path_traversable_bev[i, 1],
                                             xi=xi, yi=yi,
                                             xj=xj, yj=yj)
                    plt.pause(.1)  # pause a bit so that plots are updated

                    theta_control = pid_steer.control(
                        value_goal=thetai,
                        value_actual=thetaj,
                        time_delta=time_delta
                    )  # this is the amount to rotate from the intermediate theta(j) to the next theta(j+1)

                    thetaj_current = (thetaj_current + theta_control)
                    agent, sim, collided = apply_velocity(
                        vel_control=vel_control,
                        agent=agent,
                        sim=sim,
                        velocity=velocity_control,
                        steer=-theta_control,  # opposite y axis
                        time_step=time_delta
                    )  # will add velocity control once steering is working
                    if enable_logging:
                        log_control(
                            xi=xi, yi=yi, thetai=thetai,
                            xj=xj, yj=yj, thetaj=thetaj,
                            thetaj_current=thetaj_current,
                            theta_control=theta_control,
                            theta_error=theta_error,
                            distance_error=distance_error
                        )
                    # update pose
                    xj, yj, thetaj = compute_pose_estimate(
                        thetaj_current=thetaj_current,
                        velocity=velocity_control,
                        xj=xj,
                        yj=yj,
                    )

                    distance_error = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    theta_error = thetai - thetaj
                    break
    sim.close()


if __name__ == "__main__":
    go_robot()
