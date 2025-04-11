import os
import numpy as np
from pathlib import Path
import yaml
import torch
import cv2
from datetime import datetime
import habitat_sim

import logging

logger = logging.getLogger("[Task Setup]")  # logger level is explicitly set below by LOG_LEVEL

from libs.experiments import model_loader
from libs.path_finding import plot_utils
from libs.control.robohop import control_with_mask

from libs.commons import utils_viz, utils_data
from libs.utils import setup_sim_plots, build_intrinsics, apply_velocity, robohop_to_pixnav_goal_mask, has_collided
from libs.logger.visualizer import Visualizer

from libs.logger.level import LOG_LEVEL

logger.setLevel(LOG_LEVEL)

import utils
from utils_sim_traj import get_pathlength_GT, find_shortest_path, get_agent_rotation_from_two_positions


class Episode:
    def __init__(self, args, path_episode, scene_name_hm3d, path_results_folder, preload_data={}):
        if args is None:
            args = get_default_args()
        self.args = args
        self.steps = 0  # only used when running real in remote mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.path_episode = path_episode
        self.path_episode_results = path_results_folder / self.path_episode.parts[-1]
        self.path_episode_results.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running {self.path_episode=}...")
        self.scene_name_hm3d = scene_name_hm3d
        self.preload_data = preload_data
        if args.env == 'sim':
            if not (self.path_episode / 'agent_states.npy').exists():
                raise FileNotFoundError(
                    f'{self.path_episode / "agent_states.npy"} does not exist...')
        else:
            self.agent_states = None
            self.final_goal_position = None
            self.traversable_class_indices = None

        # map params
        self.graph_instance_ids, self.graph_path_lengths = None, None

        # data params
        # TODO: multiple hfov variables
        self.fov_deg = self.args.sim["hfov"] if np.isin(['robohop', 'tango'], self.args.method.lower()).any() else 79
        self.hfov_radians = np.pi * self.fov_deg / 180

        # experiment params
        self.success_status = 'exceeded_steps'
        self.distance_to_goal = np.nan
        self.step_real_complete = True

        # controller params
        self.time_delta = 0.1
        self.theta_control = np.nan
        self.velocity_control = 0.05 if np.isin(['robohop', 'tango'], self.args.method.lower()).any() else np.nan
        self.pid_steer_values = [.25, 0, 0] if self.args.method.lower(
        ) == 'tango' else []
        self.discrete_action = -1

        self.image_height, self.image_width = self.args.sim["height"], self.args.sim["width"]
        self.sim, self.agent, self.distance_to_final_goal = None, None, np.nan
        if args.env == 'sim':
            self.setup_sim_agent()
            self.ready_agent()
        self.map_graph = None

        self.set_controller()

        # setup visualizer
        self.vis_img_default = np.zeros(
            (self.image_height, self.image_width, 3)).astype(np.uint8)
        self.vis_img = self.vis_img_default.copy()
        self.video_cfg = {"savepath": str(self.path_episode_results / 'repeat.mp4'),
                          "codec": 'mp4v', "fps": 6}
        self.vis = Visualizer(self.sim, self.agent,
                              self.scene_name_hm3d, env=self.args.env)
        if self.args.env == 'sim':
            self.vis.draw_teach_run(self.agent_states)

    def setup_sim_agent(self) -> tuple:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

        # get the scene
        update_nav_mesh = False
        args_sim = self.args.sim
        self.sim, self.agent, vel_control = utils.get_sim_agent(
            self.scene_name_hm3d,
            update_nav_mesh,
            width=args_sim["width"],
            height=args_sim["height"],
            hfov=args_sim["hfov"],
            sensor_height=args_sim["sensor_height"]
        )
        self.sim.agents[0].agent_config.sensor_specifications[1].normalize_depth = True

        # create and configure a new VelocityControl structure
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.controlling_ang_vel = True
        vel_control.ang_vel_is_local = True
        self.vel_control = vel_control

        self.traversable_class_indices, self.bad_goal_classes, self.cull_instance_ids = get_semantic_filters(
            self.sim, self.args.traversable_class_names)

    def ready_agent(self):
        # get the initial agent state for this episode (i.e. the starting pose)
        path_agent_states = self.path_episode / 'agent_states.npy'
        self.agent_states = np.load(
            str(path_agent_states),
            allow_pickle=True
        )
        self.agent_positions_in_map = np.array([s.position for s in self.agent_states])
        # set the final goal state for this episode
        self.final_goal_position = self.agent_states[-1].position
        self.final_goal_image_idx = len(self.agent_states) - 1

        # set the start state and set the agent to this pose
        start_state = select_starting_state(
            self.sim,
            self.args,
            self.agent_states,
            self.final_goal_position
        )
        self.agent.set_state(start_state)  # set robot to this pose
        self.start_position = start_state.position

        # define measure of success
        self.distance_to_final_goal = find_shortest_path(
            self.sim,
            p1=self.start_position,
            p2=self.final_goal_position
        )[0]

    # def set_goal_generator(self):
    #     goal_source = self.args.goal_source.lower()
    #     if goal_source == 'topological':
    #         segmentor_name = self.args.segmentor.lower()
    #
    #         self.segmentor = self.preload_data['segmentor']
    #         cfg_goalie = self.args.goal_gen
    #         cfg_goalie.update(
    #             {"use_gt_localization": self.args.use_gt_localization})
    #         if segmentor_name == 'sam2':
    #             assert cfg_goalie['matcher_name'] == 'sam2', 'TODO: is other matcher implemented for this segmentor?'
    #             cfg_goalie.update({"sam2_tracker": self.segmentor})
    #
    #         self.goal_object_id, goalNodeIdx = None, None
    #
    #         self.goalie = goal_gen.Goal_Gen(
    #             W=self.image_width,
    #             H=self.image_height,
    #             G=self.map_graph,
    #             map_path=str(self.path_episode),
    #             poses=self.agent_states,
    #             task_type=self.args.task_type,
    #             cfg=cfg_goalie
    #         )
    #
    #         if self.args.use_gt_localization:
    #             self.goalie.localizer.localizedImgIdx, _ = self.get_GT_closest_map_img()
    #
    #         # to save time over storage
    #         if not self.goalie.planner_g.precomputed_allPathLengths_found and cfg_goalie[
    #             'rewrite_graph_with_allPathLengths']:
    #             logger.info("Rewritng graph with allPathLengths")
    #             allPathLengths = self.map_graph.graph.get('allPathLengths', {})
    #             allPathLengths.update(
    #                 {cfg_goalie['edge_weight_str']: self.goalie.planner_g.allPathLengths})
    #             self.map_graph.graph['allPathLengths'] = allPathLengths
    #             pickle.dump(self.map_graph, open(self.path_graph, "wb"))
    #
    #     elif goal_source == 'gt_metric':
    #         # map_graph is not needed
    #         pass
    #     else:
    #         raise NotImplementedError(
    #             f'{self.args.goal_source=} is not defined...')

    def get_goal_object_id(self):
        if self.args.reverse:
            self.goal_object_id = utils_data.find_reverse_traverse_goal(
                self.agent, self.sim, self.final_goal_state, self.map_graph)
            if not os.path.exists(f"{self.path_episode}/reverse_goal.npy"):
                print(
                    f"Saving reverse goal to {self.path_episode}/reverse_goal.npy")
                np.save(f"{self.path_episode}/reverse_goal.npy",
                        {"instance_id": self.goal_object_id, "agent_state": self.final_goal_state})

        elif self.args.task_type == 'alt_goal':
            self.goal_object_id = utils_data.get_goal_info_alt_goal(
                self.path_episode)[-1]
        else:
            self.goal_object_id = int(str(self.path_episode).split('_')[-2])

    def set_controller(self):
        control_method = self.args.method.lower()
        goal_controller = None
        self.collided = None

        # select the type of controller to use
        if control_method == 'tango':
            from libs.control.pid import SteerPID
            from libs.control.tango import TangoControl

            pid_steer = SteerPID(
                Kp=self.pid_steer_values[0],
                Ki=self.pid_steer_values[1],
                Kd=self.pid_steer_values[2]
            )

            intrinsics = build_intrinsics(
                image_width=self.image_width,
                image_height=self.image_height,
                field_of_view_radians_u=self.hfov_radians,
                device=self.device
            )

            goal_controller = TangoControl(
                traversable_classes=self.traversable_class_indices,
                pid_steer=pid_steer,
                default_velocity_control=self.velocity_control,
                h_image=self.image_height,
                w_image=self.image_width,
                intrinsics=intrinsics,
                time_delta=self.time_delta,
                grid_size=0.125,
                device=self.device
            )

        elif control_method == 'pixnav':
            from libs.pixnav.policy_agent import Policy_Agent
            from libs.pixnav.constants import POLICY_CHECKPOINT

            goal_controller = Policy_Agent(model_path=POLICY_CHECKPOINT)
            self.collided = False

        self.goal_controller = goal_controller

    def get_GT_closest_map_img(self):
        dists = np.linalg.norm(self.agent_positions_in_map - self.agent.get_state().position, axis=1)
        topK = 2 * self.args.goal_gen['loc_radius']
        closest_idxs = np.argsort(dists)[:topK]
        # approximately subsample ref indices
        closest_idxs = sorted(closest_idxs)[::self.args.goal_gen['subsample_ref']]
        closest_idx = np.argmin(dists)
        return closest_idx, closest_idxs

    def get_goal(self, rgb, depth, semantic_instance):
        self.goal_mask = None
        _, plsDict, self.goal_mask = get_pathlength_GT(
            self.sim,
            self.agent,
            depth,
            semantic_instance,
            self.final_goal_position,
            None,
        )
        self.control_input_robohop = semantic_instance

    def get_control_signal(self, step, rgb, depth):
        control_method = self.args.method.lower()
        goals_image = None

        if control_method == 'robohop':  # the og controller
            self.velocity_control, self.theta_control, goals_image = control_with_mask(
                self.control_input_robohop,
                self.goal_mask,
                v=self.velocity_control,
                gain=1,
                tao=5
            )
            self.theta_control = -self.theta_control
            self.vis_img = (
                (255.0 - 255 * (utils_viz.goal_mask_to_vis(goals_image, outlier_min_val=255))).astype(np.uint8)
            )
        # the new and (hopefully) improved controller
        elif control_method == 'tango':
            self.velocity_control, self.theta_control, goals_image_ = self.goal_controller.control(
                depth,
                self.control_input_robohop,
                self.goal_mask,
                self.traversable_mask
            )
            if goals_image_ is not None:
                self.vis_img = (
                        255.0 - 255 * (utils_viz.goal_mask_to_vis(goals_image_, outlier_min_val=255))).astype(np.uint8)
            else:
                self.vis_img = self.vis_img_default.copy()

        elif control_method == 'pixnav':
            self.pixnav_goal_mask = robohop_to_pixnav_goal_mask(
                self.goal_mask, depth)
            if not (step % 63) or self.discrete_action == 0:
                self.goal_controller.reset(
                    rgb, self.pixnav_goal_mask.astype(np.uint8))
            self.discrete_action, predicted_mask = self.goal_controller.step(
                rgb, self.collided)
        else:
            raise NotImplementedError(
                f'{self.args.method} is not available...')
        return goals_image

    def execute_action(self):
        control_method = self.args.method.lower()

        if control_method == 'pixnav':
            action_dict = {
                0: 'stop',
                1: 'move_forward',
                2: 'turn_left',
                3: 'turn_right',
                4: 'look_up',
                5: 'look_down'
            }
            previous_state = self.agent.state
            action = action_dict[self.discrete_action]
            _ = self.sim.step(action)
            current_state = self.agent.state
            self.collided = has_collided(
                self.sim, previous_state, current_state)
        else:
            self.agent, self.sim, self.collided = apply_velocity(
                vel_control=self.vel_control,
                agent=self.agent,
                sim=self.sim,
                velocity=self.velocity_control,
                steer=-self.theta_control,  # opposite y axis
                time_step=self.time_delta
            )  # will add velocity control once steering is working

    def step_real(self, rgb):
        self.step_real_complete = False
        self.get_goal(rgb, None, None)
        self.get_control_signal(self.steps, rgb, None)
        v, w = self.velocity_control, self.theta_control
        self.log_results(self.steps)
        self.steps += 1

        v_min, v_max, w_min, w_max = [self.args.controller[k] for k in [
            'v_min', 'v_max', 'w_min', 'w_max']]
        v = min(max(v_min, v), v_max)
        w = min(max(w_min, w), w_max)
        print(v, w)

        response_dict = {}
        response_dict["velocity_control"] = 0
        response_dict["theta_control"] = w
        odometry_dict = {
            'velocity_x': v,
            'velocity_y': 0,
            'velocity_z': 0,
            'angular_x': 0,
            'angular_y': 0,
            'angular_z': w
        }
        response_dict["odometry"] = odometry_dict

        print("response ready")
        self.preload_data["zmq_client"].send_response(response_dict)
        print("response sent")
        self.step_real_complete = True
        return v, w

    def is_done(self):
        done = False
        current_robot_state = self.agent.get_state()  # world coordinates
        self.distance_to_goal = find_shortest_path(
            self.sim, p1=current_robot_state.position, p2=self.final_goal_position)[0]
        if self.distance_to_goal <= self.args.threshold_goal_distance:
            logger.info(
                f'\nWinner! dist to goal: {self.distance_to_goal:.6f}\n')
            self.success_status = 'success'
            done = True
        return done

    def set_logging(self):
        self.dirname_vis_episode = self.path_episode_results / 'vis'
        self.dirname_vis_episode.mkdir(exist_ok=True, parents=True)

    def log_results(self, final=False):
        if not final:
            if self.vis is not None:
                if self.args.env == 'sim':
                    self.update_vis_sim()
                else:
                    self.update_vis()

    def update_vis_sim(self):
        # if this is the first call, init video
        ratio = self.vis_img.shape[1] / self.vis.tdv.shape[1]
        if self.vis.video is None:
            # resize tdv to match the rgb image
            self.tdv = cv2.resize(
                self.vis.tdv, dsize=None, fx=ratio, fy=ratio)
            self.video_cfg['width'] = self.vis_img.shape[1]
            self.video_cfg['height'] = self.vis_img.shape[0] + \
                                       self.tdv.shape[0]
            self.vis.init_video(self.video_cfg)

        self.vis.draw_infer_step(self.agent.get_state())
        self.tdv = cv2.resize(
            self.vis.tdv, dsize=None, fx=ratio, fy=ratio)
        combined_img = np.concatenate(
            (self.tdv, self.vis_img), axis=0)
        self.vis.save_video_frame(combined_img)

    def update_vis(self):
        # if this is the first call, init video
        if self.vis.video is None:
            self.video_cfg['width'] = self.vis_img.shape[1]
            self.video_cfg['height'] = self.vis_img.shape[0]
            self.vis.init_video(self.video_cfg)

        self.vis.save_video_frame(self.vis_img)

    def init_plotting(self):
        # TODO: better handle 'plt' (check with SP)
        import matplotlib
        import matplotlib.pyplot as plt
        # matplotlib.use('Qt5Agg')  # need it on my environment on my system ------
        if self.args.save_vis:
            matplotlib.use('Agg')  # Use the Agg backend to suppress plots

        import matplotlib.style as mplstyle
        mplstyle.use('fast')
        mplstyle.use(['dark_background', 'ggplot', 'fast'])
        fig, ax = setup_sim_plots()
        return ax, plt

    def plot(self, ax, plt, step, rgb, depth, semantic_instance):
        goals_image = None
        semantic_instance_vis = semantic_instance

        goal_mask_vis = utils_viz.goal_mask_to_vis(self.goal_mask)
        goal = (self.goal_mask == self.goal_mask.min())
        if self.args.method.lower() == 'pixnav':
            goal += (self.pixnav_goal_mask /
                     self.pixnav_goal_mask.max()).astype(int) * 2

        plot_utils.plot_sensors(
            ax=ax,
            display_img=rgb,
            semantic=semantic_instance_vis,
            depth=depth,
            goal=goal,
            goal_mask=goal_mask_vis,
            flow_goal=goals_image if goals_image is not None else np.zeros(
                rgb.shape[:2]),
            trav_mask=self.traversable_mask,
        )

        if self.args.method.lower() == 'tango':
            plot_utils.plot_path_points(
                ax=[ax[1, 2], ax[1, 0]],
                points=self.goal_controller.point_poses,
                cost_map_relative_bev=self.goal_controller.planning_cost_map_relative_bev_safe,
                colour='red'
            )

        if self.args.save_vis:
            plt.tight_layout()
            plt.savefig(self.dirname_vis_episode /
                        f'{step:04d}.jpg', bbox_inches='tight', pad_inches=0)
        else:
            plt.pause(.05)  # pause a bit so that plots are updated

    def close(self):
        if self.vis is not None:
            self.vis.close()
        if self.sim is not None:
            self.sim.close()


def load_run_list(args, path_episode_root) -> list:
    if args.run_list == '':
        path_episodes = sorted(path_episode_root.glob('*'))
    else:
        path_episodes = []
        if args.path_run == '':
            raise ValueError('Run path must be specified when using run list!')
        if args.run_list.lower() in ['winners', 'failures', 'no_good', 'custom']:
            if args.run_list.lower() not in ['no_good', 'custom']:
                logger.info(
                    f'Setting logging to False when running winner or failure list! - arg.log_robot:{args.log_robot}')
                args.log_robot = False
            with open(str(Path(args.path_run) / 'summary' / f'{args.run_list.lower()}.csv'), 'r') as f:
                for line in f.readlines():
                    path_episodes.append(
                        path_episode_root / line[:line.rfind(f'_{args.method}')].strip('\n'))
        else:
            raise ValueError(f'{args.run_list} is not a valid option.')
    return path_episodes


def init_results_dir_and_save_cfg(args, default_logger=None):
    path_results = Path(args.path_results)
    task_str = args.task_type
    path_results_folder = (path_results / task_str / args.exp_name / args.split / args.max_start_distance /
                           f'{datetime.now().strftime("%Y%m%d-%H-%M-%S")}_{args.method.lower()}_gt_metric')
    path_results_folder.mkdir(exist_ok=True, parents=True)
    if default_logger is not None:
        default_logger.update_file_handler_root(
            path_results_folder / 'output.log')
    print(f'Logging to: {str(path_results_folder)}')
    return path_results_folder


def preload_models(args):
    segmentor = None
    if args.segmentor == 'fast_sam':
        # use predefined traversable classes with fast_sam predictions only if it is tango (tango) and infer_traversable is True
        traversable_class_names = args.traversable_class_names if (
                args.infer_traversable and args.control_method.lower() == 'tango'
        ) else None

        segmentor = model_loader.get_segmentor(
            args.segmentor,
            args.sim["width"],
            args.sim["height"],
            path_models=args.path_models,
            traversable_class_names=traversable_class_names
        )

    depth_model = None
    if args.infer_depth:
        depth_model = model_loader.get_depth_model()

    # collect preload data that each episode instance can reuse
    preload_data = {"segmentor": segmentor, "depth_model": depth_model}
    return preload_data


def set_start_state_reverse_orientation(agent_states, start_index):
    start_state = agent_states[start_index]
    # compute orientation, looking at the next GT forward step
    lookat_index = start_index - 1
    if lookat_index < 0:
        raise ValueError('Cannot reverse orientation at the start of the episode.')
    # search/validate end_idx in reverse direction
    for k in range(lookat_index, -1, -1):
        # keep looking if agent hasn't moved
        if np.linalg.norm(start_state.position - agent_states[k].position) <= 0.1:
            continue
        else:
            lookat_index = k
            break
    # looking in the reverse direction
    start_state.rotation = get_agent_rotation_from_two_positions(
        start_state.position, agent_states[lookat_index].position)
    return start_state


def closest_state(sim, agent_states, distance_threshold: float, final_position=None):
    distances = np.zeros_like(agent_states)
    final_position = agent_states[-1].position if final_position is None else final_position
    for i, p in enumerate(agent_states):
        distances[i] = find_shortest_path(sim, final_position, p.position)[0]
    start_index = ((distances - distance_threshold) ** 2).argmin()
    return start_index


def select_starting_state(sim, args, agent_states, final_position=None):
    # reverse traverse episodes end 1m before the original start, offset that
    distance_threshold_offset = 1 if args.reverse else 0
    if args.max_start_distance.lower() == 'easy':
        start_index = closest_state(
            sim, agent_states, 3 + distance_threshold_offset, final_position)
    elif args.max_start_distance.lower() == 'hard':
        if args.task_type == 'via_alt_goal':
            distance_threshold_offset += 3

        start_index = closest_state(
            sim, agent_states, 5 + distance_threshold_offset, final_position)
    elif args.max_start_distance.lower() == 'full':
        start_index = 0 if not args.reverse else len(agent_states) - 1
    else:
        raise NotImplementedError(
            f'max start distance: {args.max_start_distance} is not an available start.')
    start_state = agent_states[start_index]
    return start_state


def save_dict(full_save_path, config_dict):
    with open(full_save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def get_semantic_filters(sim, traversable_class_names):
    # setup is/is not traversable and which goals are banned (for the simulator runs)
    instance_index_to_name_map = utils.get_instance_index_to_name_mapping(
        sim.semantic_scene)
    traversable_class_indices = instance_index_to_name_map[:, 0][
        np.isin(instance_index_to_name_map[:, 1], traversable_class_names)]
    traversable_class_indices = np.unique(
        traversable_class_indices).astype(int)
    bad_goal_categories = ['ceiling', 'ceiling lower']
    bad_goal_cat_idx = instance_index_to_name_map[:, 0][
        np.isin(instance_index_to_name_map[:, 1], bad_goal_categories)]
    bad_goal_classes = np.unique(bad_goal_cat_idx).astype(int)

    cull_categories = [
        'floor', 'floor mat', 'floor vent', 'carpet', 'rug', 'doormat', 'shower floor', 'pavement', 'ground',
        'ceiling', 'ceiling lower',
    ]
    cull_instance_ids = (instance_index_to_name_map[:, 0][
        np.isin(instance_index_to_name_map[:, 1], cull_categories)]).astype(int)
    return traversable_class_indices, bad_goal_classes, cull_instance_ids


def dict_to_args(cfg_dict):
    args = type('', (), {})()
    for k, v in cfg_dict.items():
        setattr(args, k, v)
    return args


def get_default_args():
    args_dict = {
        'method': 'tango',
        'goal_source': 'gt_metric',
        'graph_filename': None,
        'max_start_distance': 'easy',
        'threshold_goal_distance': 0.5,
        'debug': False,
        'max_steps': 500,
        'run_list': '',
        'path_run': '',
        'path_models': None,
        'log_robot': True,
        'save_vis': False,
        'plot': False,
        'infer_depth': False,
        'infer_traversable': False,
        'segmentor': 'fast_sam',
        'task_type': 'original',
        'use_gt_localization': False,
        'env': 'sim',
        'goal_gen': {
            'textLabels': [],
            'matcher_name': 'lightglue',
            'map_matcher_name': 'lightglue',
            'geometric_verification': True,
            'match_area': False,
            'goalNodeIdx': None,
            'edge_weight_str': None,
            'use_goal_nbrs': False,
            'plan_da_nbrs': False,
            'rewrite_graph_with_allPathLengths': False,
            'loc_radius': 4,
            'do_track': False,
            'subsample_ref': 1,
        },
        'sim': {
            'width': 320,
            'height': 240,
            'hfov': 120,
            'sensor_height': 0.4
        },
    }
    # args_dict as attributes of args
    return dict_to_args(args_dict)
