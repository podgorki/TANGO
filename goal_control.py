import os
import utils
import pickle
import torch
import numpy as np
import nav_parser
import habitat_sim
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
import logging
import time
import yaml
from libs.control.pid import SteerPID
from libs.control.tango import TangoControl
from libs.control.robohop import control_with_mask
from libs.depth.depth_anything_metric_model import DepthAnythingMetricModel
from libs.goal_generator import goal_gen
import libs.path_finding.plot_utils as plot_utils
from libs.segmentor import sam, fast_sam_module
from libs.utils import split_observations, build_intrinsics, apply_velocity, setup_sim_plots, \
    robohop_to_pixnav_goal_mask, has_collided, initialize_results, write_results, write_final_meta_results, \
    get_traversibility
from libs.utils_goals import getGoalMask, find_graph_instance_ids_and_path_lengths, change_edge_attr
from utils_sim_traj import get_pathlength_GT, find_shortest_path

## pixnav
from libs.pixnav.policy_agent import Policy_Agent
from libs.pixnav.constants import *

import matplotlib

matplotlib.use('Agg')
torch.backends.cudnn.benchmark = True

logging.basicConfig(filename=f"./out/nav_{time.time()}.log",
                    filemode='a',
                    format='%(asctime)s %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger("[Goal Control]")


def load_run_list(args, path_episode_root) -> list:
    if args.run_list == '':
        episodes = sorted(path_episode_root.glob('*'))
    else:
        episodes = []
        if args.path_run == '':
            raise ValueError('Run path must be specified when using run list!')
        if args.run_list.lower() in ['winners', 'failures', 'no_good']:
            print(f'Setting logging to False when running winner or failure list! - arg.log_robot:{args.log_robot}')
            args.log_robot = False
            with open(str(Path(args.path_run) / 'summary' / f'{args.run_list.lower()}.csv'), 'r') as f:
                for line in f.readlines():
                    episodes.append(path_episode_root / line[:line.rfind('_robohop')].strip('\n'))
        else:
            raise ValueError(f'{args.run_list} is not a valid option.')
    return episodes


def setup_sim(path_scene: Path, method) -> tuple:
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    # get args
    args_sim = nav_parser.parse_args("-ds hm3d -v 0.05 -c 'ceiling' -d".split(" "))
    seed = args_sim.seed
    np.random.seed(seed)

    # get the scene
    update_nav_mesh = False
    test_scene = str(sorted(path_scene.glob('*basis.glb'))[0])
    sim, agent, vel_control, sim_settings = utils.get_sim_agent(
        test_scene=test_scene, updateNavMesh=update_nav_mesh, method=method
    )
    sim.agents[0].agent_config.sensor_specifications[1].normalize_depth = True

    # create and configure a new VelocityControl structure
    vel_control = habitat_sim.physics.VelocityControl()
    vel_control.controlling_lin_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.controlling_ang_vel = True
    vel_control.ang_vel_is_local = True

    return sim, agent, vel_control, sim_settings


def closest_state(sim, agent_states, distance_threshold: float):
    distances = np.zeros_like(agent_states)
    final_position = agent_states[-1].position
    for i, p in enumerate(agent_states):
        distances[i] = find_shortest_path(sim, final_position, p.position)[0]
    start_index = ((distances - distance_threshold) ** 2).argmin()
    return agent_states[start_index]


def select_starting_state(sim, args, agent_states):
    if args.max_start_distance.lower() == 'easy':
        starting_state = closest_state(sim, agent_states, 3)
    elif args.max_start_distance.lower() == 'hard':
        starting_state = closest_state(sim, agent_states, 5)
    elif args.max_start_distance.lower() == 'full':
        starting_state = agent_states[0]
    else:
        raise NotImplementedError(f'max start distance: {args.max_start_distance} is not an available start.')
    return starting_state


def run(args):
    # set up all the paths
    path_dataset = Path(args.path_dataset)  # can use this to glob the episodes for evaluation
    path_scenes_root = path_dataset / 'hm3d_v0.2' / args.split
    path_episode_root = path_dataset / f'hm3d_iin_{args.split}'
    print(f'Root path for episodes: {path_episode_root}')
    if args.log_robot:
        path_results = Path(args.path_results)
        path_results_folder = (path_results / args.split / args.max_start_distance /
                               f'{datetime.now().strftime("%Y%m%d-%H-%M-%S")}_{args.method.lower()}_{args.goal_source}')
        path_results_folder.mkdir(exist_ok=True, parents=True)
        print(f'Logging to: {str(path_results_folder)}')

    episodes = load_run_list(args, path_episode_root)  # [args.sidx:args.eidx]
    device = 'cuda'

    if args.infer_depth:
        depth_model_name = 'zoedepth'
        path_zoe_depth = Path.cwd() / 'model_weights' / 'depth_anything_metric_depth_indoor.pt'
        if not path_zoe_depth.exists():
            raise FileNotFoundError(f'{path_zoe_depth} not found...')
        depth_model = DepthAnythingMetricModel(depth_model_name, pretrained_resource=str(path_zoe_depth))

    for ei, episode in tqdm(enumerate(episodes)):
        # set these now incase an episode fails
        success_status = 'exceeded_steps'
        step = np.nan
        time_delta = 0.1
        theta_control = np.nan
        velocity_control = 0.05 if 'robohop' in args.method.lower() else np.nan
        pid_steer_values = [.25, 0, 0] if args.method.lower() == 'robohop+' else []
        discrete_action = -1
        distance_to_goal = np.nan
        scene_name = episode.parts[-1].split('_')[0]
        print(episode)
        path_scene = sorted(path_scenes_root.glob(f'*{scene_name}'))[0]
        # set up path for results
        if args.log_robot:
            filename_metadata_episode = path_results_folder / f'{episode.parts[-1]}_{args.method.lower()}_{args.goal_source}.txt'
            filename_results_episode = path_results_folder / f'{episode.parts[-1]}_{args.method.lower()}_{args.goal_source}.csv'
        try:
            # setup sim, experiment etc
            max_steps = args.max_steps
            sim, agent, vel_control, sim_settings = setup_sim(path_scene, args.method)
            fov_deg = sim_settings['hfov']
            hfov_radians = np.pi * fov_deg / 180
            print(sim_settings)
            # setup is/is not traversable and which goals are banned (for the simulator runs)
            sem_insta_2_cat_map = utils.get_instance_to_category_mapping(sim.semantic_scene)
            if args.goal_source.lower() == 'topological':
                traversable_categories = [
                    'floor', 'flooring', 'floor mat', 'floor vent', 'carpet', 'mat', 'rug', 'doormat',
                    'shower floor', 'pavement', 'ground', 'tiles'
                ]
            else:
                traversable_categories = [
                    'floor', 'flooring', 'floor mat', 'floor vent', 'carpet', 'mat', 'rug', 'doormat',
                    'shower floor', 'pavement', 'ground', 'tiles'
                ]
            traversable_cat_idx = utils.get_instance_index_to_name_mapping(sim.semantic_scene)[:, 0][
                np.isin(utils.get_instance_index_to_name_mapping(sim.semantic_scene)[:, 1], traversable_categories)]
            traversable_classes = np.unique(traversable_cat_idx).astype(int)
            bad_goal_catagories = ['ceiling', 'ceiling lower']
            bad_goal_cat_idx = utils.get_instance_index_to_name_mapping(sim.semantic_scene)[:, 0][
                np.isin(utils.get_instance_index_to_name_mapping(sim.semantic_scene)[:, 1], bad_goal_catagories)]
            bad_goal_classes = np.unique(bad_goal_cat_idx).astype(int)

            if args.plot:
                import matplotlib
                import matplotlib.pyplot as plt
                matplotlib.use('Qt5Agg')  # need it on my environment on my system ------
                import matplotlib.style as mplstyle
                mplstyle.use('fast')
                mplstyle.use(['dark_background', 'ggplot', 'fast'])
                fig, ax = setup_sim_plots()
            # set up scene
            print(episode)
            # get the initial agent state for this episode (i.e. the starting pose)
            path_agent_states = episode / 'agent_states.npy'
            agent_states = np.load(
                str(path_agent_states),
                allow_pickle=True
            )
            start_state = select_starting_state(sim, args, agent_states)
            agent.set_state(start_state)  # set robot to this pose
            final_goal_position = agent_states[-1].position
            start_position = start_state.position
            distance_to_final_goal = find_shortest_path(sim, p1=start_position, p2=final_goal_position)[0]

            # setup controller
            image_height, image_width = sim.config.agents[0].sensor_specifications[0].resolution
            intrinsics = build_intrinsics(
                image_width=image_width,
                image_height=image_height,
                field_of_view_radians_u=hfov_radians,
                device=device
            )
            # select the goal source for ablation studies
            if args.goal_source.lower() in ['gt_topological', 'topological']:  # load robohop graph
                print(args.goal_source.lower())
                if args.goal_source.lower() == 'topological':
                    path_graph = episode / 'nodes_fast_sam_graphObject_4_lightglue.pickle'
                else:
                    path_graph = episode / 'nodes_graphObject_4.pickle'
                print(f'Loading graph: {path_graph}')
                map_graph = pickle.load(open(str(path_graph), 'rb'))
                map_graph = change_edge_attr(map_graph)

                if args.goal_source.lower() == 'topological':
                    goalie = goal_gen.Goal_Gen(
                        W=image_width,
                        H=image_height,
                        G=map_graph,
                        map_path=str(episode),
                        task_type='original',
                        poses=agent_states
                    )
                    if args.segmentor.lower() == 'sam':
                        raise NotImplementedError(f'{args.segmentor} not implemented...')
                        # segmentor = sam.Seg_SAM(
                        #     args.path_models, device,
                        #     resize_w=image_width,
                        #     resize_h=image_height
                        # )
                    elif args.segmentor.lower() == 'fastsam':
                        segmentor = fast_sam_module.FastSamClass(
                            config_settings={
                                'width': image_width,
                                'height': image_height,
                                'mask_height': image_height,
                                'mask_width': image_width,
                                'conf': 0.5,
                                'model': 'FastSAM-s.pt',
                                'imgsz': int(max(image_height, image_width, 480))
                            },
                            device=torch.device(
                                "cuda") if torch.cuda.is_available() else torch.device(
                                "cpu"),
                            traversable_categories=traversable_categories
                        )  # imgsz < 480 gives poorer results
                    elif args.segmentor.lower() == 'sim':
                        raise ValueError('Simulator segments not supported in topological mode...')
                    else:
                        raise NotImplementedError(f'{args.segmentor} not implemented...')
                else:
                    goal_object_id = int(str(episode).split('_')[-2])
                    graph_instance_ids, graph_path_lengths = find_graph_instance_ids_and_path_lengths(
                        map_graph,
                        goal_object_id,
                        device=device
                    )

            # select the type of controller to use
            if args.method.lower() == 'robohop+':
                pid_steer = SteerPID(
                    Kp=pid_steer_values[0],
                    Ki=pid_steer_values[1],
                    Kd=pid_steer_values[2]
                )
                goal_controller = TangoControl(
                    traversable_classes=traversable_classes,
                    pid_steer=pid_steer,
                    default_velocity_control=velocity_control,
                    h_image=image_height,
                    w_image=image_width,
                    intrinsics=intrinsics,
                    time_delta=time_delta,
                    grid_size=0.125,
                    device=device
                )
            elif args.method.lower() == 'pixnav':
                policy_agent = Policy_Agent(model_path=POLICY_CHECKPOINT)
                collided = False

            goal_position = agent_states[-1].position  # goal
            # setup the result output files
            if args.log_robot:
                initialize_results(
                    filename_metadata_episode,
                    filename_results_episode,
                    args,
                    pid_steer_values,
                    hfov_radians,
                    time_delta,
                    velocity_control,
                    goal_position,
                    traversable_classes
                )
            for step in range(max_steps):
                current_robot_state = agent.get_state()  # world coordinates
                distance_to_goal = find_shortest_path(sim, p1=current_robot_state.position, p2=final_goal_position)[0]
                if distance_to_goal <= args.threshold_goal_distance:
                    print(f'\nWinner! dist to goal: {distance_to_goal:.6f}\n')
                    success_status = 'success'
                    break
                observations = sim.get_sensor_observations()
                display_img, depth, semantic_instance = split_observations(
                    observations,
                    sem_insta_2_cat_map,
                )
                if args.goal_source.lower() == 'gt_metric':
                    _, _, goal_mask = get_pathlength_GT(
                        sim,
                        agent,
                        depth,
                        semantic_instance,
                        goal_position,
                        None
                    )
                    control_input_robohop = semantic_instance
                elif args.goal_source.lower() == 'gt_topological':
                    goal_mask = getGoalMask(
                        G_insta_ids=graph_instance_ids,
                        pls=graph_path_lengths,
                        sem=semantic_instance,
                        device=device
                    )
                    control_input_robohop = semantic_instance
                    # remove the naughty masks
                    goal_mask[np.isin(semantic_instance, bad_goal_classes)] = 99
                elif args.goal_source.lower() == 'topological':
                    semantic_instance_seg, _, traversable_mask = segmentor.segment(display_img[:, :, :3],
                                                                                   retMaskAsDict=True)
                    goal_mask = goalie.get_goal_mask(
                        qryImg=display_img[:, :, :3],
                        qryNodes=semantic_instance_seg,
                        qryPosition=current_robot_state.position if args.debug else None)
                    control_input_robohop = [goalie.pls, goalie.coords]
                else:
                    raise NotImplementedError(f'{args.goal_source} is not available...')
                if not args.infer_traversable:  # override the FastSAM traversable mask
                    traversable_mask = get_traversibility(
                        torch.from_numpy(semantic_instance),
                        traversable_classes
                    ).numpy()
                else:
                    semantic_instance = semantic_instance_seg

                if args.infer_depth:
                    depth = depth_model.infer(display_img)  # * 0.44  # is a scaling factor

                # do the control with the selected method
                if args.method.lower() == 'robohop':  # the og controller
                    velocity_control, theta_control, goals_image = control_with_mask(
                        control_input_robohop,
                        goal_mask,
                        v=velocity_control,
                        gain=1,
                        tao=5
                    )
                    theta_control = -theta_control
                elif args.method.lower() == 'robohop+':  # the new and (hopefully) improved controller
                    velocity_control, theta_control = goal_controller.control(
                        depth,
                        control_input_robohop,
                        goal_mask,
                        traversable_mask
                    )
                elif args.method.lower() == 'pixnav':
                    pixnav_goal_mask = robohop_to_pixnav_goal_mask(goal_mask, depth)
                    if not (step % 63) or discrete_action == 0:
                        policy_agent.reset(display_img, pixnav_goal_mask.astype(np.uint8))
                    discrete_action, predicted_mask = policy_agent.step(display_img, collided)
                else:
                    raise NotImplementedError(f'{args.method} is not available...')
                if args.goal_source.lower() == 'topological':
                    # convert dict to mask
                    new_semantic_instance = np.zeros(display_img.shape[:2])
                    for n in range(len(semantic_instance_seg)):
                        new_semantic_instance += semantic_instance_seg[n]["segmentation"] * n
                    semantic_instance = new_semantic_instance
                if args.plot:
                    goal = (goal_mask == goal_mask.min()).astype(int)
                    if args.method.lower() == 'pixnav':
                        goal += (pixnav_goal_mask / pixnav_goal_mask.max()).astype(int) * 2
                    plot_utils.plot_sensors(
                        ax=ax,
                        display_img=display_img,
                        semantic=semantic_instance,
                        depth=depth,
                        relative_bev=goal_controller.bev_relative if args.method.lower() == 'robohop+' else None,
                        goal=goal,
                        goal_mask=goal_mask,
                        flow_goal=goals_image if args.method.lower() == 'robohop' else None,
                    )
                    if args.method.lower() == 'robohop+':
                        plot_utils.plot_path_points(
                            ax=[ax[1, 2], ax[1, 0]],
                            points=goal_controller.point_poses,
                            cost_map_relative_bev=goal_controller.planning_cost_map_relative_bev_safe,
                            colour='red'
                        )
                        plot_utils.plot_position(axs=[ax[1, 2], ax[1, 0]],
                                                 x_image=0,
                                                 y_image=0,
                                                 xi=goal_controller.xi, yi=goal_controller.yi,
                                                 xj=goal_controller.xj, yj=goal_controller.yj)
                    plt.pause(.05)  # pause a bit so that plots are updated
                # execute action from the chosen method
                if args.method.lower() == 'pixnav':
                    action_dict = {
                        0: 'stop',
                        1: 'move_forward',
                        2: 'turn_left',
                        3: 'turn_right',
                        4: 'look_up',
                        5: 'look_down'
                    }
                    previous_state = agent.state
                    action = action_dict[discrete_action]
                    _ = sim.step(action)
                    current_state = agent.state
                    collided = has_collided(sim, previous_state, current_state)
                else:
                    agent, sim, collided = apply_velocity(
                        vel_control=vel_control,
                        agent=agent,
                        sim=sim,
                        velocity=velocity_control,
                        steer=-theta_control,  # opposite y axis
                        time_step=time_delta
                    )  # will add velocity control once steering is working
                if args.log_robot:
                    write_results(
                        filename_results_episode,
                        step,
                        current_robot_state,
                        distance_to_goal,
                        velocity_control,
                        theta_control,
                        collided,
                        discrete_action
                    )

            if args.plot:
                plt.close()
            sim.close()
        except Exception as e:
            try:  # in case the sim is on
                sim.close()
            except Exception:
                pass  # in case it is off
            success_status = e

        if args.log_robot:
            write_final_meta_results(
                filename_metadata_episode=filename_metadata_episode,
                success_status=success_status,
                final_distance=distance_to_goal,
                step=step,
                distance_to_final_goal=distance_to_final_goal
            )
        print(f'Completed with success status: {success_status}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--debug",
                        help="For debugging the localiser and goal mask generator", action='store_true')
    parser.add_argument("--log_robot",
                        help="To log the robots position etc or not", action='store_true')
    parser.add_argument("--plot",
                        help="To plot the robotic observations and derived information", action='store_true')
    parser.add_argument("--path_dataset", type=str, help="Path to the dataset root")
    parser.add_argument("--split",
                        type=str, default='train',
                        help="Split to evaluate. Default=train")
    parser.add_argument("--path_results", type=str, help="Path to the folder for recording results")
    parser.add_argument("--path_models", type=str, help="Path to the models")

    parser.add_argument("--max_steps",
                        type=int, help="Maximum simulator steps before episode ends. Default=1000", default=500)
    parser.add_argument("--method",
                        type=str, default='sim',
                        help="Method to evaluate. "
                             "Available options: [robohop, robohop+, pixnav]. Default=robohop+")
    parser.add_argument("--segmentor",
                        type=str, default='robohop+',
                        help="Method to evaluate. "
                             "Available options: [sim, sam, fastsam]. Default=sim")
    parser.add_argument("--goal_source",
                        type=str, default='gt_metric',
                        help="Source of the goal information. "
                             "Available options: [gt_metric, gt_topological]. Default=gt_metric")
    parser.add_argument("--threshold_goal_distance",
                        type=float, help="Threshold for deciding if we have arrived at the goal. Default=1m",
                        default=1.)
    parser.add_argument("--max_start_distance",
                        type=str, help="Max starting distance for the robot. "
                                       "Available options ['easy', 'hard', 'full'], where easy=3m, hard=5 "
                                       "and full is the maximum distance for that episode.",
                        default='full')
    parser.add_argument("--infer_depth",
                        help="Use depth anything metric to infer depth.", action='store_true')
    parser.add_argument("--infer_traversable",
                        help="Use CLIP to infer traversable segments. Must use FastSAM", action='store_true')
    parser.add_argument("--run_list",
                        type=str, default='',
                        help="Iterate over a predefined csv listing. "
                             "The csv should be in the results/{episode}/summary/ folder. "
                             "Choices: ['', 'winners', 'failures', 'no_good']. Default='' (no list)")
    parser.add_argument("--path_run",
                        type=str, default='',
                        help="Run path whose list you want to look at. Default='' (none).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # source = ['gt_metric'] #, 'gt_topological', 'gt_metric', 'gt_topological']
    # max_start_distance = ['full'] #, 'easy', 'hard', 'hard']
    # for s, d in zip(source, max_start_distance):
    #     args.goal_source = s
    #     args.max_start_distance = d
    #     print(f'Running with args: {args}')

    config_file = "configs/robohop.yaml"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Config File {config_file} params: {config}")
            # pass the config to the args
            for k, v in config.items():
                setattr(args, k, v)

    run(args)
