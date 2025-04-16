import os
import cv2
import torch
import logging
import habitat_sim
import numpy as np
from pathlib import Path
from datetime import datetime

from src.plotting import plot_utils, utils_visualize
from src.tango.robohop.controller import control_with_mask
from src.utils import setup_sim_plots, build_intrinsics, apply_velocity, robohop_to_pixnav_goal_mask, has_collided
from src.common import utils_data, utils, third_party_model_loader
from src.common.utils_sim_traj import get_pathlength_GT, find_shortest_path
from src.logger.visualizer import Visualizer
from src.logger.level import LOG_LEVEL

logger = logging.getLogger("[Task Setup]")  # logger level is explicitly set below by LOG_LEVEL
logger.setLevel(LOG_LEVEL)


class Episode:
    def __init__(self, args, path_episode, scene_name_hm3d, path_results_folder, preload_data={}):
        if args is None:
            args = utils.get_default_args()
        self.args = args
        self.steps = 0  # only used when running real in remote mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.path_episode = path_episode
        self.path_episode_results = path_results_folder / self.path_episode.parts[-1]
        self.path_episode_results.mkdir(parents=True, exist_ok=True)

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

        # data params
        is_robohop_tango =  np.isin(['robohop', 'tango'], self.args.method.lower()).any()
        self.fov_deg = self.args.sim["hfov"] if is_robohop_tango else 79
        self.hfov_radians = np.pi * self.fov_deg / 180

        # experiment params
        self.success_status = 'exceeded_steps'
        self.distance_to_goal = np.nan
        self.step_real_complete = True

        # controller params
        self.time_delta = 0.1
        self.theta_control = np.nan
        self.velocity_control = 0.05 if is_robohop_tango else np.nan
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
        self.vis_img_default = np.zeros((self.image_height, self.image_width, 3)).astype(np.uint8)
        self.vis_img = self.vis_img_default.copy()
        self.video_cfg = {
            "savepath": str(self.path_episode_results / 'repeat.mp4'),
            "codec": 'mp4v',
            "fps": 6
        }
        self.vis = Visualizer(self.sim, self.agent, self.scene_name_hm3d, env=self.args.env)
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
            self.sim, self.args.traversable_class_names
        )

    def ready_agent(self):
        # get the initial agent state for this episode (i.e. the starting pose)
        path_agent_states = self.path_episode / 'agent_states.npy'
        self.agent_states = np.load(
            str(path_agent_states),
            allow_pickle=True
        )
        self.agent_positions_in_map = np.array([s.position for s in self.agent_states])
        # set the final goal state for this episode
        self.final_goal_state = None
        self.final_goal_position = None
        self.final_goal_image_idx = None

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

    def get_goal_object_id(self):
        if self.args.reverse:
            self.goal_object_id = utils_data.find_reverse_traverse_goal(
                self.agent, self.sim, self.final_goal_state, self.map_graph)
            if not os.path.exists(f"{self.path_episode}/reverse_goal.npy"):
                print(f"Saving reverse goal to {self.path_episode}/reverse_goal.npy")
                np.save(
                    f"{self.path_episode}/reverse_goal.npy",
                    {"instance_id": self.goal_object_id, "agent_state": self.final_goal_state}
                )
        else:
            self.goal_object_id = int(str(self.path_episode).split('_')[-2])

    def set_controller(self):
        control_method = self.args.method.lower()
        goal_controller = None
        self.collided = None

        # select the type of controller to use
        if control_method == 'tango':
            from src.tango.pid import SteerPID
            from src.tango.tango import TangoControl

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
            from third_party.pixnav.policy_agent import Policy_Agent
            from third_party.pixnav.constants import POLICY_CHECKPOINT

            goal_controller = Policy_Agent(model_path=POLICY_CHECKPOINT)
            self.collided = False

        self.goal_controller = goal_controller

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
                (255.0 - 255 * (utils_visualize.goal_mask_to_vis(goals_image, outlier_min_val=255))).astype(np.uint8)
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
                        255.0 - 255 * (utils_visualize.goal_mask_to_vis(goals_image_, outlier_min_val=255))).astype(np.uint8)
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
            )  # will add velocity robohop once steering is working

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
            print(f'\nWinner! dist to goal: {self.distance_to_goal:.6f}\n')
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

        goal_mask_vis = utils_visualize.goal_mask_to_vis(self.goal_mask)
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
            plt.savefig(self.dirname_vis_episode / f'{step:04d}.jpg', bbox_inches='tight', pad_inches=0)
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

        segmentor = third_party_model_loader.get_segmentor(
            args.segmentor,
            args.sim["width"],
            args.sim["height"],
            path_models=args.path_models,
            traversable_class_names=traversable_class_names
        )

    depth_model = None
    if args.infer_depth:
        depth_model = third_party_model_loader.get_depth_model()

    # collect preload data that each episode instance can reuse
    preload_data = {"segmentor": segmentor, "depth_model": depth_model}
    return preload_data


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