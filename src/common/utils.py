import numpy as np
from typing import Optional
import habitat_sim
from habitat_sim.utils import common as sim_utils
import torch
from PIL import Image


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
            'text_labels': [],
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


def get_sim_settings(
        scene: str,
        default_agent: int = 0,
        sensor_height: float = 1.5,
        width: int = 256,
        height: int = 256,
        hfov: float = 90.
) -> dict:
    sim_settings = {
        "scene": scene,  # Scene path
        "default_agent": default_agent,  # Index of the default agent
        "sensor_height": sensor_height,  # Height of sensors in meters, relative to the agent
        "width": width,  # Spatial resolution of the observations
        "height": height,
        "hfov": hfov
    }
    return sim_settings


# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    if "scene_dataset_config_file" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    else:
        annotConfigPath = find_annotation_path(settings["scene"])
        if annotConfigPath is not None:
            print(f"Annotation file found: {annotConfigPath}")
            sim_cfg.scene_dataset_config_file = annotConfigPath
        else:
            print(f"Annotation file not found for {settings['scene']}")

    # agent
    hardware_config = habitat_sim.agent.AgentConfiguration()
    # # Modify the attributes you need
    # hardware_config.height = 20  # Setting the height to 1.6 meters
    # hardware_config.radius = 10  # Setting the radius to 0.2 meters
    # discrete actions defined for objectnav task in habitat-lab/habitat/config/habitat/task/objectnav.yaml
    custom_action_dict = {
        'stop': habitat_sim.ActionSpec(name='move_forward', actuation=habitat_sim.ActuationSpec(amount=0))
    }
    for k in hardware_config.action_space.keys():
        custom_action_dict[k] = hardware_config.action_space[k]
    custom_action_dict['look_up'] = habitat_sim.ActionSpec(
        name='look_up',
        actuation=habitat_sim.ActuationSpec(amount=30)
    )
    custom_action_dict['look_down'] = habitat_sim.ActionSpec(
        name='look_down',
        actuation=habitat_sim.ActuationSpec(amount=30)
    )

    hardware_config.action_space = custom_action_dict
    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    # add depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.hfov = settings["hfov"]

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.hfov = settings["hfov"]

    hardware_config.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [hardware_config])


def get_sim_agent(
        test_scene: str,
        width: int = 320,
        height: int = 240,
        hfov: float = 90.,
        sensor_height: float = 1.5
):
    sim_settings = get_sim_settings(
        scene=test_scene,
        width=width,
        height=height,
        hfov=hfov,
        sensor_height=sensor_height
    )
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    sim.pathfinder.seed(42)
    agent_state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(agent_state)

    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

    return sim, agent, action_names


def get_K_from_parameters(hfov_degree, width, height):
    hfov = np.deg2rad(float(hfov_degree))
    K = np.array([
        [(width / 2.) / np.tan(hfov / 2.), 0., width / 2.],
        [0., (height / 2.) / np.tan(hfov / 2.), height / 2.],
        [0., 0., 1]])
    return K


def get_K_from_agent(agent):
    specs = agent.agent_config.sensor_specifications[0]
    return get_K_from_parameters(specs.hfov, specs.resolution[1], specs.resolution[0])


# Habitat Semantics
def find_annotation_path(scene_path):
    # find split name from among ['train', 'val', 'test', 'minival']
    split = None
    for s in ['train', 'minival', 'val', 'test']:
        if s in scene_path:
            split = s
            path_till_split = scene_path.split(split)[0]
            break
    if split is None:
        return None
    else:
        return f"{path_till_split}/{split}/hm3d_annotated_{split}_basis.scene_dataset_config.json"


def obj_id_to_int(obj):
    return int(obj.id.split("_")[-1])


def get_instance_index_to_name_mapping(semantic_scene):
    instance_index_to_name = np.array([[i, obj.category.name()] for i, obj in enumerate(semantic_scene.objects)])
    return instance_index_to_name


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


def split_observations(observations):
    rgb_obs = observations["color_sensor"]
    depth = observations["depth_sensor"]
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    display_img = np.array(rgb_img.convert('RGB'))
    semantic_instance = observations["semantic_sensor"]  # an array of instance ids
    return display_img, depth, semantic_instance.astype(int)


def robohop_to_pixnav_goal_mask(goal_mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    max_depth_indices = np.where(depth == depth[goal_mask == goal_mask.min()].max())
    indices_goal_mask = (goal_mask == goal_mask.min())[max_depth_indices]
    goal_target = np.array(max_depth_indices).T[indices_goal_mask]
    target_x = goal_target[0, 1]
    target_z = goal_target[0, 0]
    min_z = max(target_z - 5, 0)
    max_z = min(target_z + 5, goal_mask.shape[0])
    min_x = max(target_x - 5, 0)
    max_x = min(target_x + 5, goal_mask.shape[1])
    pixnav_goal_mask = np.zeros_like(goal_mask)
    pixnav_goal_mask[min_z:max_z, min_x:max_x] = 255
    return pixnav_goal_mask


def unproject_points(depth: torch.Tensor, intrinsics_inv, homogeneous_pts) -> torch.Tensor:
    unprojected_points = (torch.matmul(intrinsics_inv, homogeneous_pts)).T
    unprojected_points *= depth
    return unprojected_points


def has_collided(sim, previous_agent_state, current_agent_state):
    # Check if a collision occured
    previous_rigid_state = habitat_sim.RigidState(
        sim_utils.quat_to_magnum(previous_agent_state.rotation), previous_agent_state.position
    )
    current_rigid_state = habitat_sim.RigidState(
        sim_utils.quat_to_magnum(current_agent_state.rotation), current_agent_state.position
    )
    dist_moved_before_filter = (
            current_rigid_state.translation - previous_rigid_state.translation
    ).dot()
    end_pos = sim.step_filter(
        previous_rigid_state.translation, current_rigid_state.translation
    )
    dist_moved_after_filter = (
            end_pos - previous_rigid_state.translation
    ).dot()

    # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
    # collision _didn't_ happen. One such case is going up stairs.  Instead,
    # we check to see if the the amount moved after the application of the filter
    # is _less_ than the amount moved before the application of the filter
    EPS = 1e-5
    collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
    return collided


def get_traversibility(semantic: torch.Tensor, traversable_classes: list) -> torch.Tensor:
    return torch.isin(semantic, torch.tensor(traversable_classes)).to(int)


def apply_velocity(vel_control, agent, sim, velocity, steer, time_step):
    # Update position
    forward_vec = habitat_sim.utils.quat_rotate_vector(agent.state.rotation, np.array([0, 0, -1.0]))
    new_position = agent.state.position + forward_vec * velocity

    # Update rotation
    new_rotation = habitat_sim.utils.quat_from_angle_axis(steer, np.array([0, 1.0, 0]))
    new_rotation = new_rotation * agent.state.rotation

    # Step the physics simulation
    # Integrate the velocity and apply the transform.
    # Note: this can be done at a higher frequency for more accuracy
    agent_state = agent.state
    previous_rigid_state = habitat_sim.RigidState(
        sim_utils.quat_to_magnum(agent_state.rotation), agent_state.position
    )

    target_rigid_state = habitat_sim.RigidState(
        sim_utils.quat_to_magnum(new_rotation), new_position
    )

    # manually integrate the rigid state
    target_rigid_state = vel_control.integrate_transform(
        time_step, target_rigid_state
    )

    # snap rigid state to navmesh and set state to object/agent
    # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
    end_pos = sim.step_filter(
        previous_rigid_state.translation, target_rigid_state.translation
    )

    # set the computed state
    agent_state.position = end_pos
    agent_state.rotation = sim_utils.quat_from_magnum(
        target_rigid_state.rotation
    )
    agent.set_state(agent_state)

    # Check if a collision occurred
    dist_moved_before_filter = (
            target_rigid_state.translation - previous_rigid_state.translation
    ).dot()
    dist_moved_after_filter = (
            end_pos - previous_rigid_state.translation
    ).dot()

    # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
    # collision _didn't_ happen. One such case is going up stairs.  Instead,
    # we check to see if the the amount moved after the application of the filter
    # is _less_ than the amount moved before the application of the filter
    EPS = 1e-5
    collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
    # run any dynamics simulation
    sim.step_physics(dt=time_step)

    return agent, sim, collided