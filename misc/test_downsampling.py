# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


#%%
# test downsampling of binary array
# create a random binary array with few ones
mh,mw = 18,18
# set random seed torch
torch.manual_seed(0)
# np.random.seed(0)
masks = torch.randint(0,2,(mh,mw))
masks = []
for i in range(2):
    mask = np.zeros((mh,mw),dtype=bool)
    r = np.random.randint(0,mh)
    c = np.random.randint(0,mw)
    h = np.random.randint(2,7)
    w = np.random.randint(2,7)
    mask[r:r+h,c:c+w] = True
    masks.append(mask[np.newaxis,:,:])
    print(masks[-1].shape)
masks = torch.cat([torch.tensor(mask) for mask in masks])
masks = masks[0]
print(masks.shape)

mode = 'nearest-exact' # 'bilinear', 'nearest', 'bicubic', 'area'
# downsample the mask using max pool
masks_down_ = F.adaptive_max_pool2d(masks[None,None,:,:].float(), output_size=(mh//2,mw//2))
masks_down = F.interpolate(masks[None,None,:,:].float(), size=(mh//2,mw//2), mode=mode)

# create an upsampled masks from masks_down
# use torch.nn.MaxUnpool2d
# masks_up = torch.nn.MaxUnpool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0))(masks_down[None,None,:,:])
masks_up = F.interpolate(masks_down.float(), size=(mh,mw), mode=mode)
masks_down1 = F.adaptive_max_pool2d(masks[None,None,:,:].float(), output_size=(mh//2,mw//2))
print(masks.sum(),masks_down[0,0].sum())
# subplot three masks
fig, axes = plt.subplots(1,3)
axes[0].imshow(masks.bool())
axes[0].set_title('Original')
axes[1].imshow(masks_down[0,0])
axes[1].set_title('Downsampled')
axes[2].imshow(masks_up[0,0])
axes[2].set_title('Upsampled')
plt.show()


# %%
# create a random array of shape 22 x 500 x 500
arr = np.random.rand(22,500,500)
# create another array of shape 500 x 500 x 2
idx = np.random.randint(0,35,(500,500,2))
# np ravel multiindex 
idx_flat = np.ravel_multi_index(idx.T, (35,35))

# create a random array of shape 22 x 35 x 35 
out = np.random.rand(22,35,35)
