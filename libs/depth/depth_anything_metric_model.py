import cv2
import torch
import torch.nn as nn
import numpy as np
import kornia as K
from pathlib import Path
from libs.depth.depth_anything.metric_depth.zoedepth.utils.config import get_config
from libs.depth.depth_anything.metric_depth.zoedepth.models.builder import build_model


class DepthAnythingMetricModel(nn.Module):

    def __init__(self, model_name, pretrained_resource):
        super().__init__()
        self.device = 'cuda'
        config = get_config(model_name, "infer", None)
        config.pretrained_resource = "local::" + str(Path(__file__).parents[0] / pretrained_resource)
        self.depth_anything_model = build_model(config).to(self.device).eval()

    @torch.inference_mode()
    def infer(self, image: np.array):
        """
        :param image:
        :param info: List containing [Fx, Fy, Cx, Cy]
        :return:
        """
        h, w, c = image.shape
        image_tensor = K.image_to_tensor(image)[None, ...].to(self.device)

        depth = self.depth_anything_model(image_tensor / 255.)  # added float right now to get it going

        if isinstance(depth, dict):
            depth = depth.get('metric_depth', depth.get('out'))
        elif isinstance(depth, (list, tuple)):
            depth = depth[-1]
        depth = K.geometry.transform.resize(depth, (h, w), interpolation='nearest', align_corners=None, side='short',
                                            antialias=False)
        return K.tensor_to_image(depth)
