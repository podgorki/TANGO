import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Optional


def value_to_colour(values, vmin=None, vmax=None, cmName='jet'):
    cmapPaths = matplotlib.cm.get_cmap(cmName)
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = np.array([cmapPaths(norm(value))[:3] for value in values])
    return colors, norm


def visualize_flow(cords_org, cords_dst, img=None, colors=None, weights=None):
    diff = cords_org - cords_dst
    dpi = 100
    img_height, img_width = img.shape[:2]  # Get the image dimensions
    fig_width, fig_height = img_width / dpi, img_height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    if img is not None: ax.imshow(img)
    if weights is not None:
        weighted_sum = (weights[:, None] * diff).sum(0)
        ax.quiver(
            *np.array([160, 120]).T,
            weighted_sum[0],
            weighted_sum[1],
            color='black',
            edgecolor='white',
            linewidth=0.5
        )
    ax.quiver(*cords_org.T, diff[:, 0], diff[:, 1], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlim([0, img_width])
    ax.set_ylim([img_height, 0])
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # return the figure as image (same size as img imshow-ed above)
    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    vis = cv2.resize(vis, (img.shape[1], img.shape[0]))
    plt.close(fig)
    return vis


def draw_masks_with_colours(im_, masks, colors, alpha=0.5):
    im = im_.copy() / 255.0
    viz = im.copy()  # np.zeros((im.shape[0],im.shape[1],3))
    for j in range(len(masks)):
        viz[masks[j]] = colors[j]
    im = alpha * viz + (1 - alpha) * im
    return im


def goal_mask_to_vis(goal_mask, outlier_min_val=99):
    """
    convert goal mask to visualisation mask
    """
    goal_mask_vis = goal_mask.copy()
    outlier_mask = goal_mask_vis >= outlier_min_val
    # if all are outliers, set all to outlier_min_val
    if np.sum(~outlier_mask) == 0:
        goal_mask_vis = outlier_min_val * np.ones_like(goal_mask_vis)
    # elif it is a mix of inliers/outliers, replace outliers with max of inliers
    elif np.sum(outlier_mask) != 0 and np.sum(~outlier_mask) != 0:
        goal_mask_vis[outlier_mask] = goal_mask_vis[~outlier_mask].max() + 1
    # invert the mask for visualisation
    goal_mask_vis = goal_mask_vis.max() - goal_mask_vis
    return goal_mask_vis


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
