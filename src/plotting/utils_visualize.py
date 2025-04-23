import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def value_to_colour(values, vmin=None, vmax=None, cmName='jet'):
    # check if colormaps has get_cmap method (newer versions of matplotlib)
    if hasattr(matplotlib.colormaps, 'get_cmap'):
        cmatch_map_masks_with_goal_masks = matplotlib.colormaps.get_cmap(cmName)
    else:
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
