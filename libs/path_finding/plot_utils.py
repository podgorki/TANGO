from typing import Optional, List
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
        relative_bev: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        goal_mask: Optional[np.ndarray] = None,
        flow_goal: Optional[np.ndarray] = None
):
    ax[0, 0].cla()
    ax[0, 0].imshow(display_img)
    ax[0, 0].set_title(f'RGB (live)')
    # Hide X and Y axes label marks
    ax[0, 0].xaxis.set_tick_params(labelbottom=False)
    ax[0, 0].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])

    ax[0, 1].cla()
    ax[0, 1].imshow(semantic)
    ax[0, 1].set_title('semantics (live)')
    # Hide X and Y axes label marks
    ax[0, 1].xaxis.set_tick_params(labelbottom=False)
    ax[0, 1].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])

    ax[0, 2].cla()
    ax[0, 2].imshow(depth)
    ax[0, 2].set_title('depth (live)')
    # Hide X and Y axes label marks
    ax[0, 2].xaxis.set_tick_params(labelbottom=False)
    ax[0, 2].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    extent = [-6, 6, 0, 10]
    ax[1, 1].cla()
    if relative_bev is not None:
        ax[1, 1].imshow(relative_bev, origin='lower', extent=extent)
        ax[1, 1].set_title('traversability est. (live)')
    else:
        if flow_goal is not None:
            ax[1, 1].imshow(flow_goal)
            ax[1, 1].set_title('flow goal. (live)')
            ax[1, 1].xaxis.set_tick_params(labelbottom=False)
            ax[1, 1].yaxis.set_tick_params(labelleft=False)
            # Hide X and Y axes tick marks
            ax[1, 1].set_xticks([])
            ax[1, 1].set_yticks([])

    goal_cmap = 'tab20c'
    if goal_mask is not None:
        goal_mask = np.clip(goal_mask, 0, 19)
        ax[0, 3].imshow(goal_mask, cmap=goal_cmap)
        ax[0, 3].xaxis.set_tick_params(labelbottom=False)
        ax[0, 3].yaxis.set_tick_params(labelleft=False)
        ax[0, 3].set_xticks([])
        ax[0, 3].set_yticks([])
        ax[0, 3].set_title('goal mask')
    ax[1, 3].clear()
    ax[1, 3].imshow(goal, extent=extent) #, cmap=goal_cmap)
    ax[1, 3].set_title('goal mask min goal')
    ax[1, 3].xaxis.set_tick_params(labelbottom=False)
    ax[1, 3].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[1, 3].set_xticks([])
    ax[1, 3].set_yticks([])
    return ax


def plot_path_points(ax, points, cost_map_relative_bev, colour: str = 'blue'):
    ax[0].cla()
    ax[0].plot(points[:, 0], points[:, 1], color=colour)
    ax[0].set_title('pose-dead reckoning w/pos')
    ax[0].set_xlim([-6, 6])
    ax[0].set_ylim([0, 10])
    ax[1].cla()
    ax[1].imshow(cost_map_relative_bev, cmap='inferno', origin='lower', extent=[-6, 6, 0, 10])
    ax[1].plot(points[:, 0], points[:, 1])
    ax[1].set_title('dist field w/pos')


def plot_position(axs: list, x_image: int, y_image: int, xi: float, yi: float, xj: float, yj: float) -> None:
    # update position on projected path
    # axs[0].cla()
    axs[0].scatter(xj, yj, color='black')
    axs[0].scatter(xi, yi, color='yellow', marker='o')
    # update position on original image
    # axs[1].scatter(x_image, y_image, color='black', marker='x')
