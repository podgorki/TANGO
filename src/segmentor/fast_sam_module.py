#!/usr/bin/env python3
"""
Author:
    Lachlan Mares, lachlan.mares@adelaide.edu.au

License:
    GPL-3.0

Description:

"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

import kornia as K
from pathlib import Path
import ultralytics.models.fastsam as fastsam
from ultralytics.utils.ops import scale_masks
from PIL import Image


class FastSamClass:
    def __init__(self, config_settings: dict, device: str = 'cuda', traversable_categories: list = None):
        root_dir = Path(__file__).resolve().parents[2]
        model_dir = root_dir / 'third_party' / 'models'
        self.device = device
        self.image_width = config_settings["width"]
        self.image_height = config_settings["height"]
        self.mask_height = config_settings["mask_height"]
        self.mask_width = config_settings["mask_width"]
        self.no_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        overrides = dict(conf=config_settings["conf"],
                         task="segment",
                         mode="predict", model=str(model_dir / config_settings["model"]),
                         save=False,
                         verbose=False,
                         imgsz=config_settings["imgsz"])

        self.predictor = fastsam.FastSAMPredictor(overrides=overrides)
        _ = self.predictor(np.zeros((config_settings["height"], config_settings["width"], 3), dtype=np.uint8))
        self.traversable_categories = traversable_categories if traversable_categories is not None else []

    @torch.inference_mode()
    def infer(self, image: np.array):
        results = self.predictor(image)[0]
        return self.process_results(results=results)

    @torch.inference_mode()
    def segment(self, image: np.array, return_mask_as_dict: bool = True, text_labels: list = []) -> tuple:
        results = self.predictor(image)[0]
        if results.masks is None:
            print(f"No masks found")
            return None, None, self.no_mask
        if len(self.traversable_categories) != 0:
            traversable_results, _, _ = self.prompt([results], texts=self.traversable_categories)
        else:
            traversable_results = []

        if len(text_labels) > 0:
            _, text_results_idx, text_sim = self.prompt(results, texts=text_labels)
            results = results[~text_results_idx[0]]

        mask_data = results.masks.data
        # Re-order based upon mask size
        mask_sums = torch.sum(mask_data, dim=(1, 2), dtype=torch.int32)

        # some masks are all zeros
        valid_masks_index = torch.argwhere(mask_sums != 0).squeeze(dim=1)
        mask_sums = mask_sums[valid_masks_index]
        mask_data = mask_data[valid_masks_index]

        ordered_sums = torch.argsort(mask_sums, descending=True).to(torch.int32)

        traversable_mask = torch.zeros_like(results[0].masks.data)
        if len(traversable_results) > 0:
            for m in traversable_results:
                traversable_mask += m.masks.data.sum(0).to(int)

        interpolated_tensor = F.interpolate(
            torch.concatenate((mask_data[ordered_sums], traversable_mask))[None, ...],
            size=(self.image_height, self.image_width),
            mode="nearest-exact"
        )[0]
        traversable_mask = interpolated_tensor[-1, ...].to(torch.bool)
        traversable_mask = K.tensor_to_image(traversable_mask)

        masks = interpolated_tensor[:-1, ...].to(torch.bool).cpu().numpy()
        mask_sums = mask_sums[ordered_sums].cpu().numpy()

        if return_mask_as_dict:
            return [{'segmentation': masks[i], 'area': mask_sums[i]} for i in
                    range(masks.shape[0])], None, traversable_mask

        else:
            return masks, mask_sums, traversable_mask

    def infer_with_points(self, image: np.array, points: list, add_points_masks: bool):
        results = self.predictor(image)[0]
        results_mask = self.process_results(results=results)
        point_masks = []

        for point in points:
            point_mask = self.process_results(self.predictor.prompt(results, points=point)[0])
            if add_points_masks:
                results_mask[point_mask > 0] = np.max(results_mask) + 1
            point_masks.append(point_mask)

        return results_mask, np.asarray(point_masks)

    def infer_with_boxes(self, image: np.array, boxes: list, add_boxes_masks: bool):
        results = self.predictor(image)[0]
        results_mask = self.process_results(results=results)
        box_masks = []

        for box in boxes:
            box_mask = self.process_results(self.predictor.prompt(results, bboxes=box)[0])
            if add_boxes_masks:
                results_mask[box_mask > 0] = np.max(results_mask) + 1
            box_masks.append(box_mask)

        return results_mask, np.asarray(box_masks)

    def infer_seg_hop(self, image: np.array, points: list):
        results = self.predictor(image)[0]

        # Append point query results
        for point in points:
            predictor_results = self.predictor.prompt(results, points=point)[0]
            results.masks.data = torch.cat((results.masks.data, predictor_results.masks.data), dim=0)

        # Re-order based upon mask size
        mask_sums = torch.argsort(torch.sum(results.masks.data, dim=(1, 2)), descending=True).to(torch.int32)

        results.masks.data = results.masks.data[mask_sums].to(torch.uint8).unsqueeze(0)
        results.masks.data = F.interpolate(results.masks.data, size=(self.image_height, self.image_width),
                                           mode="nearest-exact").squeeze()

        results_stack = results.masks[0].data.squeeze()
        masks_tensor = results_stack.clone()
        results_shape = (results.masks.data.shape[0], self.image_height, self.image_width)

        for i in range(1, results_shape[0]):
            result = results.masks[i].data.squeeze()
            results_stack = torch.cat((results_stack, results.masks[i].data.squeeze()), dim=0)
            masks_tensor[result > 0] = i + 1

        non_zeros_indices = torch.nonzero(results_stack).cpu().numpy()

        return cv2.resize(masks_tensor.cpu().numpy(), (self.image_width, self.image_height),
                          interpolation=cv2.INTER_NEAREST), non_zeros_indices, results_shape

    @torch.inference_mode()
    def process_results(self, results):
        try:
            mask_shape = results.masks[0].data.shape[1:]
            masks_tensor = torch.zeros(mask_shape, dtype=torch.uint8, device=self.device)
            # Sort the masks by size
            mask_sums = torch.argsort(torch.sum(results.masks.data, dim=(1, 2)), descending=True).to(torch.int32)

            for i, mask in enumerate(results.masks.data[mask_sums]):
                masks_tensor[mask > 0] = i + 1

            return cv2.resize(masks_tensor.cpu().numpy(), (self.image_width, self.image_height),
                              interpolation=cv2.INTER_NEAREST)

        except:
            return np.zeros((self.image_height, self.image_width), dtype=np.uint8)

    def prompt(self, results, bboxes=None, points=None, labels=None, texts=None):
        """
        Internal function for image segmentation inference based on cues like bounding boxes, points, and masks.
        Leverages SAM's specialized architecture for prompt-based, real-time segmentation.

        Args:
            results (Results | List[Results]): The original inference results from FastSAM models without any prompts.
            bboxes (np.ndarray | List, optional): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | List, optional): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | List, optional): Labels for point prompts, shape (N, ). 1 = foreground, 0 = background.
            texts (str | List[str], optional): Textual prompts, a list contains string objects.

        Returns:
            (List[Results]): The output results determined by prompts.
        """
        if bboxes is None and points is None and texts is None:
            return results
        prompt_results, idx_list, sim_list = [], [], []
        if not isinstance(results, list):
            results = [results]
        for result in results:
            masks = result.masks.data
            if masks.shape[1:] != result.orig_shape:
                masks = scale_masks(masks[None], result.orig_shape)[0]
            # bboxes prompt
            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)
            if bboxes is not None:
                bboxes = torch.as_tensor(bboxes, dtype=torch.int32, device=self.device)
                bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
                bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
                mask_areas = torch.stack([masks[:, b[1]: b[3], b[0]: b[2]].sum(dim=(1, 2)) for b in bboxes])
                full_mask_areas = torch.sum(masks, dim=(1, 2))

                union = bbox_areas[:, None] + full_mask_areas - mask_areas
                idx[torch.argmax(mask_areas / union, dim=1)] = True
            if points is not None:
                points = torch.as_tensor(points, dtype=torch.int32, device=self.device)
                points = points[None] if points.ndim == 1 else points
                if labels is None:
                    labels = torch.ones(points.shape[0])
                labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
                assert len(labels) == len(
                    points
                ), f"Excepted `labels` got same size as `point`, but got {len(labels)} and {len(points)}"
                point_idx = (
                    torch.ones(len(result), dtype=torch.bool, device=self.device)
                    if labels.sum() == 0  # all negative points
                    else torch.zeros(len(result), dtype=torch.bool, device=self.device)
                )
                for point, label in zip(points, labels):
                    point_idx[torch.nonzero(masks[:, point[1], point[0]], as_tuple=True)[0]] = True if label else False
                idx |= point_idx
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                crop_ims, filter_idx = [], []
                for i, b in enumerate(result.boxes.xyxy.tolist()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    if masks[i].sum() <= 100:
                        filter_idx.append(i)
                        continue
                    crop_ims.append(Image.fromarray(result.orig_img[y1:y2, x1:x2, ::-1]))
                similarity = self.predictor._clip_inference(crop_ims, texts)
                text_idx = torch.argmax(similarity, dim=-1)  # (M, )
                if len(filter_idx):
                    text_idx += (torch.tensor(filter_idx, device=self.device)[:, None] <= text_idx[None, :]).sum(0)
                idx[text_idx] = True
            prompt_results.append(result[idx])
            idx_list.append(idx)
            sim_list.append(similarity)

        return prompt_results, idx_list, sim_list

    def visualize(self, image: np.array, masks: np.array):
        if type(masks[0]) == dict:
            masks = [m['segmentation'] for m in masks]
        colors, _ = src.plotting.utils_visualize.value_to_colour(np.arange(len(masks)), cm_name='viridis')
        img = cv2.resize(image, (self.image_width, self.image_height))
        viz = src.plotting.utils_visualize.draw_masks_with_colours(img, masks, colors)
        plt.imshow(viz)
        plt.axis('off')
        plt.show()


# Example usage:
# python -m src.segmentor.fast_sam_module /path/to/image
if __name__ == "__main__":
    if len(sys.argv) > 1:
        imgName = sys.argv[1]
    else:
        imgName = f"{os.path.expanduser('~')}/fastdata/navigation/hm3d_iin_train/1S7LAXRdDqK_0000000_plant_42_/images/00010.png"
    img = cv2.imread(imgName)[:, :, ::-1]

    config_settings = {"conf": 0.25, "model": "FastSAM-s.pt", "imgsz": 480, "height": img.shape[0],
                       "width": img.shape[1], "mask_height": 416, "mask_width": 480}
    fastsam = FastSamClass(config_settings, 'cuda')

    # base test
    everything_results = fastsam.predictor(imgName)
    text_results = fastsam.predictor.prompt(everything_results, texts="floor")
    mask_text = text_results[0].masks.data[0].cpu().numpy()
    img_resized = cv2.resize(img, [mask_text.shape[1], mask_text.shape[0]])
    plt.imshow(img_resized)
    plt.imshow(mask_text, alpha=0.5)
    plt.show()

    # wrapper test
    masks = fastsam.segment(img, return_mask_as_dict=False, text_labels=['ceiling', 'floor'])[0]
    print(f"Found {len(masks)} masks")

    fastsam.visualize(img, masks)
