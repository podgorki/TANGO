import torch
from pathlib import Path
import importlib.resources as pkg_resources
import third_party.models

# ignore FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated.
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="contextlib")

def get_depth_model():
    from src.depth.depth_anything_metric_model import DepthAnythingMetricModel

    depth_model_name = 'zoedepth'
    with pkg_resources.path(third_party.models, 'depth_anything_metric_depth_indoor.pt') as p:
        path_zoe_depth = Path(p)  # p is a pathlib.Path object
    if not path_zoe_depth.exists():
        raise FileNotFoundError(f'{path_zoe_depth} not found...')

    depth_model = DepthAnythingMetricModel(
        depth_model_name, pretrained_resource=str(path_zoe_depth)
    )
    return depth_model


def get_segmentor(segmentor_name, image_width, image_height, device=None,
                  path_models=None, traversable_class_names=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    segmentor = None
    if segmentor_name == 'fast_sam':
        from src.segmentor import fast_sam_module

        segmentor = fast_sam_module.FastSamClass(
            config_settings={
                'width': image_width,
                'height': image_height,
                'mask_height': image_height,
                'mask_width': image_width,
                'conf': 0.5,
                'model': 'FastSAM-s.pt',
                'imgsz': int(max(image_height, image_width, 480))
            },
            device=device, traversable_categories=traversable_class_names
        )  # imgsz < 480 gives poorer results

    elif segmentor_name == 'sim':
        raise ValueError('Simulator segments not supported in topological mode...')

    else:
        raise NotImplementedError(f'{segmentor_name=} not implemented...')

    return segmentor
