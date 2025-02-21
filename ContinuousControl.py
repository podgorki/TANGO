import numpy as np
import random

import habitat_sim
from habitat_sim.utils import common as utils

class ContinuousControl():
    def __init__(self, control_frequency=5, frame_skip=12):
        """
        control_frequency: int, default=5. Control frequency for the agent. 
        frame_skip: int, default=12. Number of frames to skip while controlling the agent. 
        """
        self.control_frequency = control_frequency
        self.frame_skip = frame_skip
        self.fps = self.control_frequency * self.frame_skip
        self.time_step = 1.0 / self.frame_skip # In the example code, this is 1/fps, should be 1/frame_skip?

        # create and configure a new VelocityControl structure
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True

    def act(self,fwdVel,rotVel,agent,sim):
        # update the velocity control
        # local forward is -z
        self.vel_control.linear_velocity = np.array([0, 0, -fwdVel])
        # local up is y
        self.vel_control.angular_velocity = np.array([0, rotVel, 0])

        # simulate and collect frames
        observations = []
        for _frame in range(self.frame_skip):
            obs, collided = self.apply_continuous_velocity(agent, sim)
            observations.append(obs)
        return observations

    # manually control the object's kinematic state via velocity integration
    def apply_continuous_velocity(self, agent, sim):
        # Integrate the velocity and apply the transform.
        # Note: this can be done at a higher frequency for more accuracy
        agent_state = agent.state
        previous_rigid_state = habitat_sim.RigidState(
            utils.quat_to_magnum(agent_state.rotation), agent_state.position
        )

        # manually integrate the rigid state
        target_rigid_state = self.vel_control.integrate_transform(
            self.time_step, previous_rigid_state
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
        sim.step_physics(self.time_step)


        # render observation
        observation = sim.get_sensor_observations()

        return observation, collided

    def generate_random_continuous_action_sequence(self, sim_time):
        control_sequence = []
        for _action in range(int(sim_time * self.control_frequency)):
            # allow forward velocity and y rotation to vary
            cont_action_rand = self.generate_random_continuous_actions()
            control_sequence.append(cont_action_rand)
        return control_sequence

    def generate_random_continuous_actions(self):
        cont_action_rand = [
            random.random() * 2.0,  # [0,2),
            (random.random() - 0.5) * 2.0,  # [-1,1)
        ]
        return cont_action_rand

    def generate_random_continuous_actions_dict(self):
        cont_action_rand = {
                "forward_velocity": random.random() * 2.0,  # [0,2)
                "rotation_velocity": (random.random() - 0.5) * 2.0,  # [-1,1)
            }
        return cont_action_rand