import numpy as np

from utils import visualize_flow, value2color


def control_with_GT_mask(semantic, goalMask, v=0.05, gain=0.2, tao=10):
    H, W = semantic.shape
    instances = np.unique(semantic)

    coords, pls = [], []
    for i in range(len(instances)):
        points = np.argwhere(semantic == instances[i])
        coords.append(np.array(points.mean(0)))
        pls.append(goalMask[points[0, 0], points[0, 1]])
    # NOTE: Coords are in (x=c,y=r) format
    coords = np.fliplr(np.array(coords))
    pls = np.array(pls)

    plsMask = pls < 99
    if plsMask.sum() <= 1:
        return 0.0, 0.1, np.zeros((H, W, 3))
    pls = pls[plsMask]
    coords = coords[plsMask]
    outliers = goalMask >= 99
    goalMask[outliers] = pls.max()
    goalMask = pls.max() - goalMask  # to match the visImg color scheme

    weights = 1 - (pls - pls.min()) / (pls.max() - pls.min())
    weights = np.exp(tao * weights)
    weights = weights / weights.sum()

    coordsRef = coords.copy()
    coordsRef[:, 0] = W // 2
    weightedSum = (weights[:, None] * (coords - coordsRef)).sum(0)
    x_off = weightedSum[0]
    w = -x_off * gain / (W // 2)
    v = v

    colors, norm = value2color(weights, cmName='viridis')
    visImg = visualize_flow(coords, coordsRef, goalMask, colors, norm, weights, fwdVals=None, display=False,
                            colorbar=False).astype(float) / 255.0
    visImg[outliers] = [255, 255, 255]
    return v, w, visImg
