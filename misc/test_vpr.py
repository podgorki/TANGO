# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn.functional as F
from natsort import natsorted

#import func from SegmentMap
sys.path.insert(0,"../auto_agents/SegmentMap/")
sys.path.insert(0,"../auto_agents/SegmentMap/DINO/")
import func
from AnyLoc.utilities import DinoV2ExtractFeatures, VLAD

#%%
dino = DinoV2ExtractFeatures("dinov2_vitg14", 31, 'value', device='cuda',norm_descs=False)

# %%

h5file = "../out/RoboHop/AT_old.h5"
h5Data = h5py.File(h5file, 'r')
print(h5Data.keys())
# %%
def getGraphFromCords(mask_cords):
    tri = func.Delaunay(mask_cords)
    # plt.imshow(np.argmax(masks_seg,0))
    # plt.triplot(mask_cords[:,0], mask_cords[:,1], tri.simplices)
    # plt.show()
    nbrs, nbrsLists = [], []
    rft_da_nbrs = []
    for v in range(len(mask_cords)):
        nbrsList = func.getNbrsDelaunay(tri, v)
        nbrsLists.append(nbrsList)
        nbrs += nbrsList
    return np.array(nbrs)

def vlad_matmuls_per_cluster(num_c,masks,res,clus_labels,adjMat=None,device='cuda'):
    """
    Expects input tensors to be cuda and float/double
    """
    vlads = []
    num_m = len(masks)
    if adjMat is None:
        adjMat = torch.eye((num_m,num_m),dtype=masks.dtype,device=masks.device)
    for li in range(num_c):
        inds_li = torch.where(clus_labels==li)[0]
        masks_nbrAgg = (adjMat @ masks[:,inds_li])
        vlad = masks_nbrAgg.bool().to(masks.dtype) @ res[inds_li,:]
        vlad = F.normalize(vlad, dim=1)
        vlads.append(vlad)
    vlads = torch.stack(vlads).permute(1,0,2).reshape(len(masks),-1)
    vlads = F.normalize(vlads, dim=1)
    return vlads

def name2Ft(imPath,dinoModel):
    img_p = cv2.imread(imPath)
    # plt.imshow(img_p)
    # plt.show()
    ft = func.getAnyLocFt(img_p, dinoModel,upsample=False)
    ft /= torch.linalg.norm(ft,axis=1,keepdims=True)
    return img_p, ft
# %%

S = np.array(func.getMasks_h5(h5file,list(h5Data.keys())[0])[1])
coords = np.array(([np.array(np.nonzero(s)).mean(1)[::-1].astype(int) for s in S]))
S = S.reshape(S.shape[0],-1)
print(f"Segment Masks shape: {S.shape}")
print(f"Segment Coords shape: {coords.shape}")

nbrs = getGraphFromCords(coords)
A = np.zeros((len(coords),len(coords)))
A[nbrs[:,0],nbrs[:,1]] = 1
A[nbrs[:,1],nbrs[:,0]] = 1
# plt.imshow(A)
# plt.show()

# vlad = vlad_matmuls_per_cluster(num_clusters,masks_low.float().cuda(),residuals.float().cuda(),labels)

# %%
imgDir = "/home/sourav/workspace/data/inputs/VPR/AmsterTime/old/"
imageNames = natsorted(os.listdir(imgDir))
for i,imgName in enumerate(imageNames):
    img0, ft0 = name2Ft(f'{imgDir}/{imgName}',dino)

# %%
