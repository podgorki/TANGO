import numpy as np
from src.common import utils_visualize


def control_with_mask(
        pls_coords_or_semantic, goal_mask,
        v: float = 0.05, gain: float = 0.2, tao: int = 10):
    height, width = goal_mask.shape

    if isinstance(pls_coords_or_semantic, np.ndarray):
        semantic = pls_coords_or_semantic
        instances = np.unique(semantic)

        coords, pls = [], []
        for i in range(len(instances)):
            points = np.argwhere(semantic == instances[i])
            coords.append(np.array(points.mean(0)))
            pls.append(goal_mask[points[0, 0], points[0, 1]])
        # NOTE: Coords are in (x=c,y=r) format
        coords = np.fliplr(np.array(coords))
        pls = np.array(pls)
    else:
        pls, coords = pls_coords_or_semantic

    pls_mask = pls < 99
    if pls_mask.sum() <= 1:
        return 0.0, 0.1, np.zeros((height, width, 3))
    pls = pls[pls_mask]
    coords = coords[pls_mask]
    outliers = goal_mask >= 99
    goal_mask[outliers] = pls.max()
    goal_mask = pls.max() - goal_mask  # to match the visualize_image color scheme

    weights = np.ones_like(pls)
    if np.unique(pls).shape[0] == 1:
        print(f"same path length for all matches: {pls[0]}")
    else:
        weights = 1 - (pls - pls.min()) / (pls.max() - pls.min())
        weights = np.exp(tao * weights)
        weights = weights / weights.sum()

    coords_ref = coords.copy()
    coords_ref[:, 0] = width // 2
    weighted_sum = (weights[:, None] * (coords - coords_ref)).sum(0)
    x_off = weighted_sum[0]
    w = -x_off * gain / (width // 2)
    v = v

    colors, norm = utils_visualize.value_to_colour(weights, cm_name='viridis')
    visualize_image = utils_visualize.visualize_flow(
        coords, coords_ref, goal_mask, colors, weights
    ).astype(float) / 255.0
    visualize_image[outliers] = [255, 255, 255]
    return v, w, visualize_image
