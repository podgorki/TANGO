import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from natsort import natsorted
import magnum as mn
import cv2

from habitat.utils.visualizations import maps

import utils
# import common.utils_sim_traj as ust


class Visualizer:
    def __init__(self, sim, agent, scene_name_hm3d, video_cfg={}, env='sim'):

        if env == 'sim':
            if sim is None or agent is None:
                sim, agent, _ = utils.get_sim_agent(scene_name_hm3d)
                print(f"Loaded sim and agent from {scene_name_hm3d}")
            self.sim = sim
            self.agent = agent
        else:
            self.sim = None
            self.agent = None

        # default tdv
        self.tdv_dims = (160, 120)
        self.tdv = 255 * np.ones((self.tdv_dims[0], self.tdv_dims[1], 3), dtype=np.uint8)

        # create a videowriter for inference runs
        self.video = None
        if video_cfg != {}:
            self.init_video(video_cfg)

    def init_video(self, video_cfg):
        self.video = cv2.VideoWriter(video_cfg['savepath'], cv2.VideoWriter_fourcc(
            *video_cfg['codec']), video_cfg['fps'], (video_cfg.get('width', self.tdv_dims[1]), video_cfg.get('height', self.tdv_dims[0])))

    def create_top_down_map(self, height=None, meters_per_pixel=0.025):
        if height is None:
            scene_bb = self.sim.get_active_scene_graph().get_root_node().cumulative_bb
            height = scene_bb.y().min

        top_down_map = maps.get_topdown_map(
            self.sim.pathfinder, height, meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        self.tdv = top_down_map
        self.tdv_dims = (self.tdv.shape[0], self.tdv.shape[1]) # (height, width)

    def sim_to_tdv(self, path_point):
        tdv_point = maps.to_grid(
            path_point[2],
            path_point[0],
            self.tdv_dims,
            pathfinder=self.sim.pathfinder,
        )
        return tdv_point

    def sim_to_tdv_path(self, path):
        return np.array([self.sim_to_tdv(point) for point in path])

    def draw_path(self, path, ax=None):
        if ax is not None:
            ax.plot(path[:, 1], path[:, 0], 'o-', markersize=5,
                    markerfacecolor='tab:blue', markeredgecolor='white')
        else:
            # change colors using cmap (normalized to num of points)
            cmap = plt.get_cmap('spring')
            for i in range(len(path) - 1):
                pt1 = (path[i][1], path[i][0])
                pt2 = (path[i + 1][1], path[i + 1][0])
                color = cmap(i / len(path))
                color_cv2 = [int(255 * c) for c in color[:3]]
                cv2.line(self.tdv, pt1, pt2, color_cv2, 2, cv2.LINE_AA)

    def draw_start(self, point, ax=None):
        if ax is not None:
            ax.plot(point[1], point[0], 'o', markersize=20,
                    markerfacecolor='tab:blue', markeredgecolor='white')
        else:
            pt = (point[1], point[0])
            markersize = 10
            markerfacecolor = (31, 119, 179) # RGB for 'tab:blue'
            markeredgecolor = (255, 255, 255)

            # draw outer border
            cv2.circle(self.tdv, pt, markersize, markeredgecolor, -1, cv2.LINE_AA)

            cv2.circle(self.tdv, pt, markersize-2,
                       markerfacecolor, -1, cv2.LINE_AA)

    def draw_goal(self, point, ax=None):
        if ax is not None:
            ax.plot(point[1], point[0], '*', markersize=20,
                    markerfacecolor='tab:green', markeredgecolor='white')
        else:
            markersize = 15
            markerfacecolor = (44, 160, 44)
            markeredgecolor = (255, 255, 255)

            # draw outer border
            cv2.drawMarker(self.tdv, [point[1], point[0]], markerfacecolor,
                           cv2.MARKER_TRIANGLE_UP, markersize, 4, cv2.LINE_AA)
            cv2.drawMarker(self.tdv, [point[1], point[0]+5], markerfacecolor,
                           cv2.MARKER_TRIANGLE_DOWN, markersize, 4, cv2.LINE_AA)

    def draw_teach_run(self, agent_states, draw_traj=False, display=False, save_path=None):
        path = np.array([s.position for s in agent_states])
        avg_floor_height = np.mean([s.position[1] for s in agent_states])

        self.create_top_down_map(avg_floor_height)
        tdv_path = self.sim_to_tdv_path(path)

        # draw path
        self.draw_path(tdv_path)
        # draw start
        self.draw_start(tdv_path[0])
        # draw goal
        self.draw_goal(tdv_path[-1])

        if draw_traj:
            # draw agent with heading
            for i, state in enumerate(agent_states):
                heading = rot_to_heading(state.rotation)
                maps.draw_agent(self.tdv, tdv_path[i], heading, agent_radius_px=5)

        if save_path is not None or display:
            plt.imshow(self.tdv)
            plt.axis('off')
        if save_path is not None:
            plt.savefig(save_path)
        if display:
            plt.show()

    def draw_infer_step(self, agent_state):
        tdv_point = self.sim_to_tdv(agent_state.position)
        heading = rot_to_heading(agent_state.rotation)
        maps.draw_agent(self.tdv, tdv_point, heading, agent_radius_px=5)

    def save_video_frame(self, rgb=None):
        if rgb is None:
            rgb = self.tdv
        self.video.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.video is not None:
            self.video.release()


def rot_to_heading(rot):
    R = np.array(mn.Quaternion(rot.imag, rot.real).to_matrix())
    # convert base to camera
    R_bc = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    R = R @ R_bc
    heading = np.arctan2(R[0, 2], R[2, 2])
    return heading


if __name__ == "__main__":
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    path_dataset = Path("./data/")
    split = "val"
    path_scenes_root_hm3d = path_dataset / 'hm3d_v0.2' / split
    path_episode_root = path_dataset / f'hm3d_iin_{split}'
    path_episode = natsorted(list(path_episode_root.glob("*")))[1]

    scene_name_hm3d = utils.get_hm3d_scene_name_from_episode_path(
        path_episode, path_scenes_root_hm3d)

    sim = None
    agent = None
    vis = Visualizer(sim, agent, scene_name_hm3d)
    agent_states = np.load(f"{path_episode}/agent_states.npy", allow_pickle=True)
    vis.draw_teach_run(agent_states, display=True)
    vis.sim.close()