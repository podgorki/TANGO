import sys
import cv2
import matplotlib.pyplot as plt


from libs.commons import utils_viz

# INSTALL: cd segment-anything; pip install -e .
from libs.segmentor.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ignore FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="contextlib")


class Seg_SAM:
    def __init__(self, modelPath, device='cuda', sam_kwargs={}, **kwargs):
        self.toRGB = kwargs.get("toRGB", False)
        self.resize_w = kwargs.get("resize_w", None)
        self.resize_h = kwargs.get("resize_h", None)
        self.mask_generator = self.load_model(modelPath, device, sam_kwargs)

    def load_model(self, modelPath, device, sam_kwargs={}):
        modelType = "vit_h" # TODO enable other options
        modelName = "sam_vit_h_4b8939.pth"
        print("Loading model from ", modelPath)
        sam = sam_model_registry[modelType](checkpoint=f"{modelPath}/{modelName}")
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam, **sam_kwargs)
        return mask_generator
    
    def segment(self, img, textLabels=[]):
        if self.toRGB: # cv2 reads in BGR
            img = img[:, :, ::-1]
        if self.resize_w is not None and self.resize_h is not None:
            img = cv2.resize(img, (self.resize_w, self.resize_h))
        masks = self.mask_generator.generate(img)
        return masks

# Example usage:
# INSTALL: cd segment-anything; pip install -e .
# python -m libs.segmentor.sam <path/to/image> <path/to/model> #sam_vit_h_4b8939.pth
if __name__ == "__main__":
    # imgName = f"{os.path.expanduser('~')}/fastdata/navigation/hm3d_iin_train/1S7LAXRdDqK_0000000_plant_42_/images/00010.png"
    # modelPath = f"{os.path.expanduser('~')}/workspace/s/sg_habitat/models/segment-anything/"

    imgName = sys.argv[1]
    modelPath = sys.argv[2]

    img = cv2.imread(imgName)[:,:,::-1]
    seg = Seg_SAM(modelPath)
    masks = seg.mask_generator.generate(img)

    print(f"Found {len(masks)} masks")
    plt.imshow(img)
    utils_viz.show_anns(masks)
    plt.axis('off')
    plt.show()
