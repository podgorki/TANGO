import torch
import numpy as np
import kornia as K
from kornia import morphology as morph
from scipy.interpolate import splrep, BSpline
from libs.path_finding.graphs import CostMapGraphNX
from libs.utils import unproject_points
from libs.control.robohop import control_with_mask


class TangoControl:

    @torch.inference_mode
    def __init__(self, pid_steer, traversable_classes, default_velocity_control: float,
                 h_image: int, w_image: int, intrinsics: torch.Tensor = torch.eye(3),
                 time_delta: float = 0.1, grid_size: float = 0.1, device: str = 'cpu'):
        self.default_velocity_control = default_velocity_control
        self.device = device
        self.traversable_classes = torch.from_numpy(traversable_classes).to(int).to(self.device)  # [25, 30, 37]

        self.time_delta = time_delta
        self.pid_steer = pid_steer
        # camera stuff
        self.intrinsics = intrinsics.to(self.device)
        self.intrinsics_inv = torch.linalg.inv(self.intrinsics)

        # perspective image
        self.h_image, self.w_image = h_image, w_image
        u = torch.arange(0, self.w_image, requires_grad=False, device=self.device)
        v = torch.arange(0, self.h_image, requires_grad=False, device=self.device)
        vs, us = torch.meshgrid(v, u)
        us = us.reshape(-1)
        vs = vs.reshape(-1)
        homogeneous_pts = torch.concatenate((
            us[..., None],
            vs[..., None],
            torch.ones(size=(us.shape[0], 1), device=us.device)
        ), dim=1).T.to(float)
        self.precomputed_unprojected_no_scale = (torch.matmul(self.intrinsics_inv, homogeneous_pts)).T

        # bev occupancy grid (x, y, z): (+-5m, +-5m, 0-10m)
        self.grid_size = grid_size
        self.grid_min = torch.tensor([-5, 0, 0], device=self.device)
        self.grid_max = torch.tensor([5, 10, 10], device=self.device)
        self.cells = ((self.grid_max - self.grid_min) / self.grid_size).to(int)
        self.w_bev, self.h_bev = self.cells[2].item(), self.cells[0].item()
        self.grid_shift = torch.tensor([self.h_bev // 2, -3], dtype=torch.long,
                                       device=self.device)  # todo: make the z roll be auto calculated
        self.start_bev = (self.w_bev // 2, 0)
        self.x_bev_range = torch.arange(
            self.grid_min[0].item(), self.grid_max[0].item(), self.grid_size
        ).round(decimals=3)
        self.z_bev_range = torch.arange(
            self.grid_min[2].item(), self.grid_max[2].item(), self.grid_size
        ).round(decimals=3)
        self.kernel_erode = torch.ones((2, 2), device=self.device)
        self.occupied_bev = torch.zeros(
            self.h_bev, self.w_bev,
            device=self.device, requires_grad=False, dtype=torch.long
        )
        self.free_bev = torch.zeros(
            self.h_bev, self.w_bev,
            device=self.device, requires_grad=False, dtype=torch.long
        )

    @torch.inference_mode
    def compute_goal_point(self, depth: torch.Tensor, goal_mask: torch.Tensor) -> torch.Tensor:
        max_depth_indices = torch.where(depth == depth[goal_mask == goal_mask.min()].max())

        indices_goal_mask = (goal_mask == goal_mask.min())[max_depth_indices]
        pixel_goal = torch.stack(max_depth_indices)[indices_goal_mask.repeat(2, 1)]

        homogeneous_pts = torch.ones(3, device=self.device)
        homogeneous_pts[0] = pixel_goal[1]
        homogeneous_pts[1] = pixel_goal[0]
        unprojected_point = unproject_points(
            depth[pixel_goal[0], pixel_goal[1]],
            intrinsics_inv=self.intrinsics_inv.float(),
            homogeneous_pts=homogeneous_pts
        )

        point_goal_bev = (unprojected_point / self.grid_size).long()[0::2] + self.grid_shift
        point_goal_bev[0] = point_goal_bev[0].clip(0, self.w_bev - 1)
        point_goal_bev[1] = point_goal_bev[1].clip(0, self.h_bev - 1)
        return point_goal_bev

    @torch.inference_mode
    def compute_relative_bev(self,
                             traversable: torch.Tensor,
                             depth: torch.Tensor) -> torch.Tensor:

        unprojected_points = self.precomputed_unprojected_no_scale * depth
        upper = unprojected_points < self.grid_max
        lower = unprojected_points > self.grid_min
        mask_in_range = torch.logical_and(lower, upper).all(1)
        unprojected_points = unprojected_points[mask_in_range]
        traversable = traversable[mask_in_range]
        xy_t_ij = torch.floor(unprojected_points / self.grid_size).to(torch.long)[:, 0::2] + self.grid_shift
        self.occupied_bev.zero_()
        self.free_bev.zero_()
        self.occupied_bev = (
                self.occupied_bev.to(torch.long).index_put_((xy_t_ij[:, 1], xy_t_ij[:, 0]),
                                                            torch.logical_not(traversable).long(),
                                                            accumulate=True) > 0
        ).int()
        self.free_bev = (
                self.free_bev.to(torch.long).index_put_((xy_t_ij[:, 1], xy_t_ij[:, 0]), traversable.long(),
                                                        accumulate=True) > 0
        ).int()
        occupancy = (self.free_bev - self.occupied_bev).clip(0, 1)
        return occupancy.float()

    @staticmethod
    def compute_point_tangents(points: np.ndarray) -> np.ndarray:
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

    def get_point_poses_numpy(self, path_traversable_bev: np.ndarray) -> np.ndarray:
        skips = 5
        traversable_bev_xs = self.x_bev_range[path_traversable_bev[:, 0]]
        traversable_bev_zs = self.z_bev_range[path_traversable_bev[:, 1]]
        if path_traversable_bev.shape[0] > skips:
            traversable_bev_xs = traversable_bev_xs[::skips]
            traversable_bev_zs = traversable_bev_zs[::skips]
        try:
            t = np.concatenate(
                (np.array([0]),
                 np.cumsum(np.diff(traversable_bev_xs, 1) ** 2 + np.diff(traversable_bev_zs, 1) ** 2))
            ) / traversable_bev_xs.shape[0]
            ti = np.linspace(0, t[-1], 20)
            tck_x = splrep(t, traversable_bev_xs, s=0)
            tck_z = splrep(t, traversable_bev_zs, s=0)
            traversable_bev_xs = BSpline(*tck_x)(ti)
            traversable_bev_zs = BSpline(*tck_z)(ti)
        except TypeError:
            pass  # sometimes things just dont go to plan so default to janky paths
        traversable_bev = np.concatenate((traversable_bev_xs[:, None], traversable_bev_zs[:, None]), axis=1)
        thetas = self.compute_point_tangents(traversable_bev)
        point_poses = np.concatenate((traversable_bev, thetas), axis=1)
        return point_poses

    def get_traversibility(self, semantic: torch.Tensor) -> torch.Tensor:
        return torch.isin(semantic, self.traversable_classes).to(int)

    def add_safety_margin(self, traversable: torch.Tensor) -> torch.Tensor:
        traversable_with_margin = morph.erosion(traversable[None, None, ...], self.kernel_erode)
        return traversable_with_margin

    @staticmethod
    def check_if_traversable(traversable_relative_bev: torch.Tensor) -> bool:
        return traversable_relative_bev.sum() > 10

    def control(self, depth: np.ndarray, robohop_control: np.ndarray, goal_mask: np.ndarray,
                traversable_mask: np.ndarray) -> float:
        velocity_control = 0.01  # gets over-written if we have TANGO
        goal_image = None
        depth = torch.from_numpy(depth).to(self.device)
        goal_mask = torch.from_numpy(goal_mask).to(self.device)
        point_goal_bev = self.compute_goal_point(depth, goal_mask)
        depth = depth.reshape(-1)[:, None].repeat(1, 3)
        # add a margin around the non-traversable objects
        traversable_relative_bev = self.compute_relative_bev(
            traversable=torch.from_numpy(traversable_mask).to(self.device).reshape(-1),
            depth=depth,
        )

        traversable_relative_bev_safe = self.add_safety_margin(traversable_relative_bev)

        # setup defaults in case path is bad
        self.bev_relative = np.zeros((20, 20), np.uint8)
        self.planning_cost_map_relative_bev_safe = np.zeros_like(self.bev_relative)
        self.xi, self.xj, self.yi, self.yj = 0, 0, 0, 0
        self.point_poses = np.array([
            [self.xi, self.yi],
            [self.xj, self.yj]
        ])
        # for extracting when plotting
        self.bev_relative = traversable_relative_bev_safe.squeeze(0, 1).cpu().numpy()
        if self.check_if_traversable(traversable_relative_bev_safe):
            # convert to occupancy (1: occupied, 0 free space)
            cost_map_relative_bev_safe = K.filters.box_blur(
                traversable_relative_bev_safe, (5, 5),  # (3, 3)
            ).squeeze(0, 1)  # soften edges to help keep robot from hitting wall

            cost_scaler = 1000
            cost_map_relative_bev_safe = (1 - cost_map_relative_bev_safe.squeeze(0, 1)) * cost_scaler + 1

            self.planning_cost_map_relative_bev_safe = cost_map_relative_bev_safe.cpu().numpy()  # for plottin

            # get the furthest Euclidean goal point
            goal_x = point_goal_bev[0].item()
            goal_y = point_goal_bev[1].item()
            goal_bev = (goal_x, goal_y)

            # find a path in the cost map
            cmg = CostMapGraphNX(
                width=self.w_bev,
                height=self.h_bev,
                cost_map=cost_map_relative_bev_safe.cpu().numpy()
            )
            path_traversable_bev = cmg.get_path(self.start_bev, goal_bev)  # [5:]
            if path_traversable_bev.shape[0] > 1:
                self.point_poses = self.get_point_poses_numpy(path_traversable_bev)
                # find the theta control signal: thetaj current pose, thetai target pose
                thetaj, thetai = self.point_poses[0, 2], self.point_poses[1, 2]
                theta_control = self.pid_steer.control(
                    value_goal=thetai,
                    value_actual=thetaj,
                    time_delta=self.time_delta
                )
                self.xi, self.xj = self.point_poses[0, 0], self.point_poses[1, 0]
                self.yi, self.yj = self.point_poses[0, 1], self.point_poses[1, 1]
                velocity_control = self.default_velocity_control
                return velocity_control, theta_control, goal_image  # this is the amount to rotate from the intermediate theta(j) to the next theta(j+1)
            # else:
            #     _, theta_control, goal_image = control_with_mask(
            #         robohop_control,
            #         goal_mask.cpu().numpy(),
            #         v=self.default_velocity_control,
            #         gain=1,
            #         # tao=5,
            #     )
            #     theta_control = -theta_control
        # else:
        _, theta_control, goal_image = control_with_mask(
            robohop_control,
            goal_mask.cpu().numpy(),
            v=self.default_velocity_control,
            gain=1,
            # tao=5,
        )
        theta_control = -theta_control
        return velocity_control, theta_control, goal_image  # this is the amount to rotate from the intermediate theta(j) to the next theta(j+1)
