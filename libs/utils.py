from pathlib import Path
import habitat_sim
from PIL import Image
import numpy as np
import torch
from typing import Optional
from habitat_sim.utils import common as utils
import logging
import matplotlib.pyplot as plt
import quaternion as qt

logger = logging.getLogger(__name__)


def setup_sim_plots():
    fig, ax = plt.subplots(2, 4)
    plt.ion()
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


def split_observations(observations, sem_insta_2_cat_map):
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
        utils.quat_to_magnum(previous_agent_state.rotation), previous_agent_state.position
    )
    current_rigid_state = habitat_sim.RigidState(
        utils.quat_to_magnum(current_agent_state.rotation), current_agent_state.position
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
        utils.quat_to_magnum(agent_state.rotation), agent_state.position
    )

    target_rigid_state = habitat_sim.RigidState(
        utils.quat_to_magnum(new_rotation), new_position
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
    agent_state.rotation = utils.quat_from_magnum(
        target_rigid_state.rotation
    )
    agent.set_state(agent_state)

    # Check if a collision occured
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


def get_traversibility(semantic: torch.Tensor, traversable_classes: list) -> torch.Tensor:
    return torch.isin(semantic, torch.tensor(traversable_classes)).to(int)


def log_control(xi: float, yi: float, thetai: float,
                xj: float, yj: float, thetaj: float,
                distance_error: float, theta_error: float,
                theta_control: float, thetaj_current: float) -> None:
    s = (f'distance error: {distance_error:.11f}, '
         f'tangent error: {(theta_error * 180 / np.pi):.11f}, '
         f'xi: {xi:.11f}, yi: {yi:.11f}, thetai: {(thetai * 180 / np.pi):.11f}, '
         f'xj: {xj:.11f}, yj: {yj:.11f}, thetaj: {(thetaj * 180 / np.pi):.11f}, '
         f'theta control: {(theta_control * 180 / np.pi):.11f}, '
         f'theta cumulative: {(thetaj_current * 180 / np.pi):.11f}')
    logger.info("%s", s)


def initialize_results(
        filename_metadata_episode,
        filename_results_episode,
        args,
        pid_steer_values,
        hfov_radians,
        time_delta,
        velocity_control,
        goal_position,
        traversable_categories
):
    # write metadata
    with open(str(filename_metadata_episode), 'w') as f:
        f.writelines(f'method={args.method}\n'
                     f'inferring_depth={args.infer_depth}\n'
                     f'goal_source={args.goal_source}\n'
                     f'max steps={args.max_steps}\n'
                     f'goal distance threshold={args.threshold_goal_distance}\n'
                     f'steer pid values={pid_steer_values}\n'
                     f'camera fov={(hfov_radians * 180 / np.pi):.2f}\n'
                     f'time_delta={time_delta}\n'
                     f'velocity_control={velocity_control}\n'
                     f'goal position={list(goal_position)}\n'
                     f'traversable categories={traversable_categories}\n')

    with open(str(filename_results_episode), 'a') as f:
        f.writelines(f'step,x,y,z,yaw,distance_to_goal,velocity_control,theta_control,discrete_action,collided\n')
    return


def write_results(filename_results_episode,
                  step,
                  current_robot_state,
                  distance_to_goal,
                  velocity_control,
                  theta_control,
                  collided,
                  discrete_action
                  ) -> None:
    with open(str(filename_results_episode), 'a') as f:
        f.writelines(f'{step},'
                     f'{current_robot_state.position[0]},'
                     f'{current_robot_state.position[1]},'
                     f'{current_robot_state.position[2]},'
                     f'{np.arccos(qt.as_rotation_matrix(current_robot_state.rotation)[0, 0]) * 180 / np.pi},'
                     f'{distance_to_goal},'
                     f'{velocity_control},'
                     f'{theta_control * 180 / np.pi},'
                     f'{discrete_action},'
                     f'{int(collided)}\n')


def write_final_meta_results(
        filename_metadata_episode: Path,
        success_status: str,
        final_distance: float,
        step: int,
        distance_to_final_goal):
    with open(str(filename_metadata_episode), 'a') as f:
        f.writelines(f'success_status={success_status}\n'
                     f'final_distance={final_distance}\n'
                     f'step={step}\n'
                     f'distance_to_final_goal_from_start={distance_to_final_goal}')
