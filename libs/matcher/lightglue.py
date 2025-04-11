
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger("[LightGlue]") # logger level is explicitly set below by LOG_LEVEL

from libs.matcher.LightGlue.lightglue.utils import resize_image, numpy_image_to_torch
from libs.matcher.LightGlue.lightglue import LightGlue, SuperPoint
from libs.matcher.LightGlue.lightglue.utils import load_image, rbd

from libs.logger.level import LOG_LEVEL
logger.setLevel(LOG_LEVEL)

from libs.commons import utils


class MatchLightGlue():
    def __init__(self, resize_w=320, resize_h=240, device='cuda', cfg={}):

        self.device = device
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.cfg = cfg
        self.match_area = self.cfg.get("match_area", False)

        self.lexor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.lmatcher = LightGlue(features='superpoint').eval().to(self.device)

    def getImg_path(self, imgPath):
        img = load_image(imgPath, resize=(
            self.resize_h, self.resize_w)).to(self.device)
        return img

    def getImg_arr(self, img):
        img = numpy_image_to_torch(resize_image(
            img, (self.resize_h, self.resize_w))[0]).to(self.device)
        return img

    def getImg(self, pathOrArr):
        if isinstance(pathOrArr, str):
            return self.getImg_path(pathOrArr)
        else:
            return self.getImg_arr(pathOrArr)

    def map_node2kp(self, kp, nodeInds):
        masks, areas = utils.nodes2key(
            nodeInds, 'segmentation'), utils.nodes2key(nodeInds, 'area')
        if masks.shape[1] != self.resize_h or masks.shape[2] != self.resize_w:
            masks = cv2.resize(masks.transpose(1, 2, 0).astype(
                float), (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            masks = masks.transpose(2, 0, 1)
        m = masks[:, kp[:, 1].astype(int), kp[:, 0].astype(int)]
        return m, areas

    def matchPair_imgWithMask(self, imSrc, imTgt, nodesSrc, nodesTgt, visualize=False, matcher=None, extractor=None, ftSrc=None, ftTgt=None):
        use_areas = self.match_area

        if ftSrc is None or visualize:
            imSrc = self.getImg(imSrc)
        if ftTgt is None or visualize:
            imTgt = self.getImg(imTgt)
        if extractor is None:
            extractor = self.lexor

        if ftSrc is None:
            ftSrc = extractor.extract(imSrc)
        if ftTgt is None:
            ftTgt = extractor.extract(imTgt)
        # remove batch dimension
        if matcher is None:
            matcher = self.lmatcher
        lmatches = rbd(matcher({'image0': ftSrc, 'image1': ftTgt}))
        count, score = lmatches['matches'].shape[0], lmatches['scores'].mean(
            0).detach().cpu().numpy()[()]
        # print([count,score,ftSrc['keypoints'].shape[1],ftTgt['keypoints'].shape[1]])
        if count == 0:
            return None
        ftSrc, ftTgt = [rbd(x)
                        for x in [ftSrc, ftTgt]]  # remove batch dimension
        kp1, kp2, matches = ftSrc['keypoints'], ftTgt['keypoints'], lmatches['matches']
        mkp1, mkp2 = kp1[matches[..., 0]].detach().cpu(
        ).numpy(), kp2[matches[..., 1]].detach().cpu().numpy()

        node2kp1, areas1 = self.map_node2kp(mkp1, nodesSrc)
        node2kp2, areas2 = self.map_node2kp(mkp2, nodesTgt)

        lmat_ij = (node2kp1[:, None] * node2kp2[None,]).sum(-1)
        matchesBool_lfm = lmat_ij.sum(1) != 0

        if use_areas:
            areaDiff_ij = areas1[:, None].astype(float) / areas2[None, :]
            areaDiff_ij[areaDiff_ij > 1] = 1 / areaDiff_ij[areaDiff_ij > 1]
            lmat_ij = lmat_ij * areaDiff_ij

        matches_ij = lmat_ij.argmax(1)

        if use_areas:
            matchesBool_area = areaDiff_ij[np.arange(
                len(matches_ij)), matches_ij] > 0.5
            matchesBool = np.logical_and(matchesBool_lfm, matchesBool_area)
        else:
            matchesBool = matchesBool_lfm

        im_lfm, im_lfm_nodes = None, None
        if visualize:
            im_lfm = np.concatenate(
                [imSrc.cpu().numpy(), imTgt.cpu().numpy()], axis=2).transpose(1, 2, 0)
            im_lfm_nodes = im_lfm.copy()
            for i in range(len(mkp1)):
                cv2.line(im_lfm, (int(mkp1[i][0]), int(mkp1[i][1])), (int(
                    mkp2[i][0] + imSrc.shape[1]), int(mkp2[i][1])), (255, 0, 0), 2, lineType=cv2.LINE_AA)
            coords_i, coords_j = utils.nodes2key(nodesSrc, 'coords')[matchesBool], utils.nodes2key(nodesTgt, 'coords')[
                matches_ij[matchesBool]]  # BUG ALERT: qry should be called differently when using nodes2key
            for i in range(matchesBool.sum()):
                cv2.line(im_lfm_nodes, (int(coords_i[i][0]), int(coords_i[i][1])), (int(
                    coords_j[i][0] + self.resize_w), int(coords_j[i][1])), (255, 0, 0), 2, lineType=cv2.LINE_AA)

        singleBestMatch = lmat_ij.max(1).argmax()
        return matchesBool, matches_ij, singleBestMatch, lmat_ij, [im_lfm, im_lfm_nodes]

    def matchPair_imgWithMask_multi(self, qryImg, refImgList, qryNodes, refNodesList, visualize=False, ftTgtList=None):
        matchPairs = []
        vizImgs = None

        imSrc = self.getImg(qryImg)
        ftSrc = self.lexor.extract(imSrc)

        resultsList = []
        for i in range(len(refImgList)):
            refImg, refNodes = refImgList[i], refNodesList[i]

            ftTgt = None
            if ftTgtList is not None:
                ftTgt = {key: torch.from_numpy(ftTgtList[i][key]).to(self.device) for key in ftTgtList[i].keys()}
            results = self.matchPair_imgWithMask(
                qryImg, refImg, qryNodes, refNodes, visualize, ftSrc=ftSrc, ftTgt=ftTgt)
            resultsList.append(results)

        for results in resultsList:
            if results is not None:
                matchesBool, matches_ij, _, _, vizImgs = results
                matchPairs.append(np.column_stack(
                    [np.argwhere(matchesBool).flatten(), matches_ij[matchesBool]]))
                if visualize:
                    plt.imshow(vizImgs[1])
                    plt.show()
            else:
                matchPairs.append(np.zeros((0, 2), int))
        return matchPairs, vizImgs, ftSrc


# Example usage, run from main repo:
# python -m libs.matcher.lightglue /path/to/image1 /path/to/image2 /path/to/sam_model
if __name__ == "__main__":
    # imgPath1 = f"{os.path.expanduser('~')}/fastdata/navigation/hm3d_iin_train/1S7LAXRdDqK_0000000_plant_42_/images/00010.png"
    # imgPath2 = f"{os.path.expanduser('~')}/fastdata/navigation/hm3d_iin_train/1S7LAXRdDqK_0000000_plant_42_/images/00013.png"
    # modelPath = f"{os.path.expanduser('~')}/workspace/s/sg_habitat/models/segment-anything/"

    imgPath1 = sys.argv[1]
    imgPath2 = sys.argv[2]
    modelPath = sys.argv[3]

    img1 = cv2.resize(cv2.imread(imgPath1)[:, :, ::-1], (320, 240))
    img2 = cv2.resize(cv2.imread(imgPath2)[:, :, ::-1], (320, 240))

    matcher = MatchLightGlue()

    from libs.segmentor import sam
    seg = sam.Seg_SAM(modelPath)

    masks1 = seg.segment(img1)
    print(f"Found {len(masks1)} masks for image 1")
    masks2 = seg.segment(img2)
    print(f"Found {len(masks2)} masks for image 2")

    matchesBool, matches_ij, singleBestMatch, _, vizImgs = matcher.matchPair_imgWithMask(
        imgPath1, imgPath2, masks1, masks2, visualize=True)
    print(f"Found {matchesBool.sum()} matches")

    plt.imshow(vizImgs[1])
    plt.show()
