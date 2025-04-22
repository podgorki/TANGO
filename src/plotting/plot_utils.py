from typing import Optional, List
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def setup_plot(start: tuple, goal: tuple, max_width: int, max_height: int):
    fig = plt.figure()
    xstart, ystart = start
    xgoal, ygoal = goal
    plt.scatter(xstart, ystart, color='blue')
    plt.scatter(xgoal, ygoal, color='green')
    plt.xlim([0, max_width])
    plt.ylim([0, max_height])
    return fig


def plot_paths(paths: List[list], start: tuple, goal: tuple, max_width: int, max_height: int,
               cost_map: Optional[np.array] = None, algorithms: Optional[list] = None):
    fig = setup_plot(start, goal, max_width, max_height)
    if cost_map is not None:
        plt.imshow(cost_map, cmap='bwr')
    for i, path in enumerate(paths):
        x0, y0 = path[0]
        xs, ys = [x0], [y0]
        for coordinate in path[1:]:
            x, y = coordinate
            xs.append(x)
            ys.append(y)
        if algorithms is not None:
            label = algorithms[i]
        else:
            label = ''
        plt.plot(xs, ys,
                 color=mcolors.XKCD_COLORS[list(mcolors.XKCD_COLORS.keys())[i]],
                 label=label)
    if algorithms is not None:
        plt.legend()
    return fig


def plot_sensors(
        ax,
        display_img: np.ndarray,
        semantic: np.ndarray,
        depth: np.ndarray,
        goal: Optional[np.ndarray] = None,
        goal_mask: Optional[np.ndarray] = None,
        flow_goal: Optional[np.ndarray] = None,
        trav_mask: Optional[np.ndarray] = None
):
    ax[0, 0].cla()
    ax[0, 0].imshow(display_img)
    ax[0, 0].set_title(f'RGB')
    # Hide X and Y axes label marks
    ax[0, 0].xaxis.set_tick_params(labelbottom=False)
    ax[0, 0].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])

    ax[0, 1].cla()
    ax[0, 1].imshow(semantic)
    ax[0, 1].set_title('Segments')
    # Hide X and Y axes label marks
    ax[0, 1].xaxis.set_tick_params(labelbottom=False)
    ax[0, 1].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])

    ax[0, 2].cla()
    ax[0, 2].imshow(depth)
    ax[0, 2].set_title('Depth')
    # Hide X and Y axes label marks
    ax[0, 2].xaxis.set_tick_params(labelbottom=False)
    ax[0, 2].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    extent = [-6, 6, 0, 10]
    ax[1, 1].cla()
    ax[1, 1].imshow(display_img, extent=extent)
    ax[1, 1].imshow(trav_mask, alpha=0.5, extent=extent, cmap='Greens')
    ax[1, 1].xaxis.set_tick_params(labelbottom=False)
    ax[1, 1].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('Traversability')
    ax_flow = ax[1, 2]
    ax_flow.cla()
    ax_flow.imshow(flow_goal, extent=extent, cmap='viridis')
    ax_flow.set_title('Fallback (RoboHop)')
    ax_flow.xaxis.set_tick_params(labelbottom=False)
    ax_flow.yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax_flow.set_xticks([])
    ax_flow.set_yticks([])

    if goal_mask is not None:
        # goal_mask = np.clip(goal_mask, 0, 19)
        ax[0, 3].imshow(display_img)
        ax[0, 3].imshow(goal_mask, alpha=0.5, cmap='viridis')
        ax[0, 3].xaxis.set_tick_params(labelbottom=False)
        ax[0, 3].yaxis.set_tick_params(labelleft=False)
        ax[0, 3].set_xticks([])
        ax[0, 3].set_yticks([])
        ax[0, 3].set_title('Goal Mask')
    goal_cmap = matplotlib.colors.ListedColormap(
        [matplotlib.cm.get_cmap('Greys')(0.0), matplotlib.cm.get_cmap('viridis')(1.0)]
    )
    ax[1, 3].cla()
    ax[1, 3].imshow(display_img, extent=extent)
    ax[1, 3].imshow(goal, alpha=0.5, extent=extent, cmap=goal_cmap)
    ax[1, 3].set_title('Selected Goal')
    ax[1, 3].xaxis.set_tick_params(labelbottom=False)
    ax[1, 3].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[1, 3].set_xticks([])
    ax[1, 3].set_yticks([])
    return ax


def plot_path_points(ax, points, cost_map_relative_bev, colour: str = 'blue'):
    ax[1].cla()
    ax[1].imshow(cost_map_relative_bev, cmap='inferno', origin='lower', extent=[-6, 6, 0, 10])
    ax[1].plot(points[:, 0], points[:, 1])
    ax[1].set_title('BEV')

