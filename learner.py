# %%
import numpy as np
import os
import sys
from tqdm import tqdm
import networkx as nx
import pickle
import cv2
import matplotlib.pyplot as plt
from natsort import natsorted

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import utils
from auto_agent import visualize_flow, value2color
sys.path.insert(0,"./auto_agents/SegmentMap/")
from auto_agents.SegmentMap import func

from train.model import controlModel

# GLOBALS
FORMAT = 'regAng'
OUTDIM = 1
NUM_CLASSES = 3
MAX_LEN = 70

MODEL_TYPE = 'mlp3'
N_H = 32
N_I = 5

accum_grad = False

# %%
def convertActions(actionLabels,numClasses=NUM_CLASSES,format=FORMAT):
    if 'class' in format:
        actionLabels = (actionLabels - np.mean(actionLabels,0)) / np.std(actionLabels,0)
        actionLabels = np.digitize(actionLabels, np.linspace(-2,2,numClasses)) - 1
        if format == 'classBoth':
            actionLabels = np.eye(numClasses*numClasses)[actionLabels[:,0]*numClasses + actionLabels[:,1]]
        elif format == 'classLin':
            actionLabels = np.eye(numClasses)[actionLabels[:,0]]
        elif format == 'classAng':
            actionLabels = np.eye(numClasses)[actionLabels[:,1]]
        else:
            raise ValueError(f'format {format} not recognized')
    elif 'disc' in format:
        if format == 'discAng':
            assert(NUM_CLASSES==3)
            labs = actionLabels[:,1].copy()
            labs2 = np.zeros_like(labs,dtype=int)
            labs2[labs<0] = 0
            labs2[labs==0] = 1
            labs2[labs>0] = 2
            actionLabels = np.eye(3)[labs2]
    elif 'reg' in format:
        if format == 'regBoth':
            pass
        elif format == 'regLin':
            actionLabels = actionLabels[:,0][:,None]
        elif format == 'regAng':
            actionLabels = actionLabels[:,1][:,None]
        else:
            raise ValueError(f'format {format} not recognized')
    else:
        raise ValueError(f'format {format} not recognized')
    return actionLabels

def loadData(mapPath,predPath):
    '''
    Example:
    imagesDataDict = learner.loadData("./out/maps/multiPointTrajs/5cdEh9F2hJL_toilet_36_20240430160015093231/","./out/runs/dump/5cdEh9F2hJL_toilet_36_20240430160015093231_20240430164947798366/controlLogs/")
    '''
    # load action labels
    actionLabels = np.load(f'{mapPath}/action_labels.npy', allow_pickle=True)
    actionLabels = convertActions(actionLabels)
    numSamples = len(actionLabels)

    # load input data
    episode = np.load(f'{predPath}/episode.npy', allow_pickle=True)[()]
    imagesData = episode['learnersDict']
    w, h = 320, 240
    assert(len(actionLabels) == len(imagesData))

    vectors = []
    for i in range(numSamples):
        pl, a_q, a_r, c_q = imagesData[i]['pl'][:,None], imagesData[i]['areas_q'][:,None], imagesData[i]['areas_r'][:,None], imagesData[i]['coords_q']
        # convert c_q to -1 to 1
        c_q = c_q / np.array([w,h]) - 0.5
        # convert areas to percentage
        a_q = a_q / (w*h)
        a_r = a_r / (w*h)
        vectors.append(np.concatenate([pl/100, a_q, a_r, c_q], axis=1))
    return vectors, actionLabels

def nodes2key(G,key):
    _key = key
    if key == 'coords':
        _key = 'segmentation'
    values = np.array([G.nodes[n][_key] for n in G.nodes()])
    if key == 'coords':
        values = np.array([np.array(np.nonzero(v)).mean(1)[::-1].astype(int) for v in values])
    return values
    
def getGoalNode(mapPath,mapMasks,img_w,img_h,nodeID_to_imgRegionIdx):
    mapName = ""
    goalImgIdx = len(os.listdir(f"{mapPath}/{mapName}/images/"))-1
    episode = np.load(f"{mapPath}/{mapName}/episode.npy", allow_pickle=True)[()]
    obs_g = np.load(f"{mapPath}/{mapName}/obs_g.npy", allow_pickle=True)[()]
    goalMaskBinary = obs_g['semantic_sensor']==int(episode.goal_object_id)
    goalMaskBinary = cv2.resize(goalMaskBinary.astype(float), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    mapNodeInds_in_goalImg = np.argwhere(nodeID_to_imgRegionIdx[:,0] == goalImgIdx).flatten()
    mapMasks = mapMasks[mapNodeInds_in_goalImg].transpose(1,2,0)
    mask_and = np.logical_and(mapMasks, goalMaskBinary[:,:,None])
    mask_or = np.logical_or(mapMasks, goalMaskBinary[:,:,None])
    iou = mask_and.sum(0).sum(0) / mask_or.sum(0).sum(0)
    goalNode = mapNodeInds_in_goalImg[iou.argmax()]
    return goalNode

def loadData_mapOnly(mapPath,graphPath):
    img_w,img_h = 320,240
    actionLabels = np.load(f'{mapPath}/action_labels.npy', allow_pickle=True)
    actionLabels = convertActions(actionLabels)
    gPath = f'{graphPath}/nodes_graphObject_4.pickle'
    if not os.path.exists(gPath) or "JptJPosx1Z6" in gPath or "xWvSkKiWQpC_tv_monitor" in gPath or "3CBBjsNkhqW_chair_22" in gPath or "ooq3SnvC79d_toilet_89" in gPath or "6imZUJGRUq4_chair_126" in gPath:
        print(f'Graph file not found: {gPath}')
        return None, None, None
    G = pickle.load(open(gPath,'rb'))
    mapMasks = nodes2key(G,'segmentation')
    nodeID_to_imgRegionIdx = nodes2key(G,'map')
    areas = nodes2key(G,'area')/(img_w*img_h)
    coords = nodes2key(G,'coords')
    goalNode = getGoalNode(mapPath,mapMasks.copy(),img_w,img_h,nodeID_to_imgRegionIdx)
    pls = dict(nx.single_target_shortest_path_length(G,goalNode))
    pls = np.array([pls[src] for src in G.nodes()])
    ft_da = G.graph['rft_da_env_arr'] # already l2-normalized
    # obtain vectors from G
    vectors = []
    nodeInds = []
    # loop over each frame in the map
    for i in range(actionLabels.shape[0]):
        nodeInds_i = np.argwhere(nodeID_to_imgRegionIdx[:,0] == i).flatten()
        pls_i = pls[nodeInds_i].copy()/100.0
        if (pls_i.max() - pls_i.min()) == 0:
            pls_i = np.zeros_like(pls_i)
        else:
            pls_i = 1 + (pls_i - pls_i.min()) / (pls_i.max() - pls_i.min())
        v = np.concatenate([coords[nodeInds_i]/np.array([img_w,img_h]) - 0.5, pls_i[:,None], areas[nodeInds_i,None],  areas[nodeInds_i,None], ft_da[nodeInds_i]], axis=1)
        vectors.append(v)
        nodeInds.append(nodeInds_i)
    masks = [mapMasks[nodeInds_i] for nodeInds_i in nodeInds]
    coords = [coords[nodeInds_i] for nodeInds_i in nodeInds]
    return vectors, actionLabels, [masks,coords]

# define a function to obtain N dimensional positional encoding for each pixel of an image of size (H,W)
def positional_encoding(H, W, N):
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    pe = np.zeros((H*W, N))
    for i in range(N):
        if i % 2 == 0:
            pe[:, i] = np.sin(2*np.pi*xx*(i//2))
        else:
            pe[:, i] = np.cos(2*np.pi*yy*(i//2))
    return pe.reshape(H, W, N)

def loadData_mapOnly_multi(mapPathDir,graphPathDir,precomDataPath='./out/learner/val.pkl'):
    if os.path.exists(precomDataPath):
        # dat = np.load(precomDataPath, allow_pickle=True)
        dat = pickle.load(open(precomDataPath,'rb'))
        actionLabels = dat['actionLabels']
        actionLabels = convertActions(actionLabels)
        try:
            vectors = dat['vectors']#[()]['v']
            return vectors, actionLabels
        except:
            pass
        
    mapNames = natsorted(os.listdir(mapPathDir))
    vectors, actionLabels = [], []
    for mapName in tqdm(mapNames):
        print(mapName)
        mapPath = f'{mapPathDir}/{mapName}'
        graphPath = f'{graphPathDir}/{mapName}'
        v,_,_ = loadData_mapOnly(mapPath,graphPath)
        a = np.load(f'{mapPath}/action_labels.npy', allow_pickle=True)

        if v is None:
            continue
        else:
            vectors.extend(v)
            actionLabels.extend(a)
    actionLabels = np.vstack(actionLabels)
    # vectors = np.array(padSeq(vectors))
    pickle.dump({'vectors':vectors,'actionLabels':actionLabels}, open(precomDataPath,'wb'))
    # np.savez(precomDataPath,vectors=vectors,actionLabels=actionLabels)
    actionLabels = convertActions(actionLabels)
    return vectors, actionLabels

def drawWeightedSegmentation(paths,imgIdx,weights):
    mapPath, graphPath = paths
    img = cv2.imread(f'{mapPath}/images/{imgIdx:05d}.png')
    img = cv2.resize(img, (320,240))
    masks, coords = loadData_mapOnly(mapPath,graphPath)[-1]
    masks, coords = masks[imgIdx], coords[imgIdx]
    colors, norm = value2color(weights,cmName='winter')
    controlImg = func.drawMasksWithColors(img,masks,colors)
    coords_r_ = coords.copy()
    coords_r_[:,0] = img.shape[1]//2
    controlImg = visualize_flow(coords,coords_r_,controlImg,colors,norm,weights,fwdVals=None,display=False,colorbar=False).astype(float) / 255.0
    return controlImg
    
class TrajVectorDataset(Dataset):
    def __init__(self, vectors, actionLabels):
        self.vectors = vectors
        self.actionLabels = actionLabels

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        v = torch.tensor(self.vectors[idx][:,:N_I], dtype=torch.float32).cuda()
        a = torch.tensor(self.actionLabels[idx], dtype=torch.float32).cuda()
        return v, a
    
def padSeq(trajVectors):
    # pad the vectors to the same length
    max_len = MAX_LEN #= max([len(v) for v in trajVectors])
    for i in range(len(trajVectors)):
        trajVectors[i] = np.pad(trajVectors[i], ((0, max_len - len(trajVectors[i])), (0, 0)), mode='constant')
    return trajVectors
    
def get_dataloader(trajVectors, actionLabels, batch_size=32):
    trajVectors = padSeq(trajVectors)
    if FORMAT == 'regAng':
        actionLabels = np.clip(actionLabels, -0.2, 0.2)
    elif 'reg' in FORMAT:
        actionLabels[:,1] = np.clip(actionLabels[:,1], -0.2, 0.2)
        actionLabels[:,0] = np.clip(actionLabels[:,0], 0, 0.2)
    dataset = TrajVectorDataset(trajVectors, actionLabels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# define a training loop
def train(model, dataloader, criterion, optimizer, num_epochs=10,modelSavePath='./out/learner/trained_models/'):
    savePath = utils.createTimestampedFolderPath(modelSavePath,"")[0]
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for i_batch, sample_batched in enumerate(dataloader):
            # forward pass
            vectors = sample_batched[0]
            labels = sample_batched[1]
            # print(vectors.shape, labels.shape)
            outputs = model(vectors)[0]
            # print(outputs)
            loss = criterion(outputs, labels)
            
            if accum_grad:
                loss.backward()
                if i_batch % 16 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')
        if (epoch+1) % 20 == 0:
            torch.save(model.state_dict(), f'{savePath}/latest.pth')
    print('Finished Training')

# define a test loop
def test(model, dataloader):
    correct = 0
    total = 0
    predictions = []
    labs = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            vectors = sample_batched[0]
            labels = sample_batched[1]
            outputs = model(vectors)[0]
            
            # compare outputs with labels as regression
            if 'reg' in FORMAT:
                total += 1
                correct += (torch.sum(torch.abs(outputs - labels)) < 0.1).item()
            # consider outputs as logits, and labels as one-hot
            else:
                total += labels.size(0)
                outputs = outputs.argmax(1)
                correct += (outputs == labels.argmax(1)).sum().item()

            predictions.append(outputs.cpu().numpy()[0])
            labs.append(labels.cpu().numpy()[0])
    print(f'Accuracy: {correct/total}')
    return np.array(predictions), np.array(labs)

# main function
def main(mapPath, predPath, modelResumePath=None, split='train'):
    # vecs, labs = loadData(mapPath, predPath)
    # vecs, labs = loadData_mapOnly(mapPath, predPath)
    vecs, labs = loadData_mapOnly_multi(mapPath, predPath,precomDataPath=f'./out/learner/{split}.pkl')
    
    bs = 1 if accum_grad else 32
    dataloader = get_dataloader(vecs, labs, batch_size=bs)
    # model = QKVAttention(input_size=N_I, hidden_size=N_H, num_heads=8, outdim=labs.shape[1]).cuda()
    model = controlModel(input_size=N_I, hidden_size=N_H, outdim=OUTDIM,format=FORMAT,model_type=MODEL_TYPE).cuda()
    print(model)
    # define regression loss, 
    if 'reg' in FORMAT:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.load_state_dict(torch.load(modelResumePath)) if modelResumePath else None
    train(model, dataloader, criterion, optimizer, num_epochs=500)
    preds, labs = test(model, dataloader)
    # if 'reg' in FORMAT:
    #     print(list(zip(preds,labs)))
    # else:
    #     print(list(zip(preds,labs.argmax(1))))

# %%
if __name__ == '__main__':
    modelResumePath = None
    if len(sys.argv) < 3:
        # mapName = "h1zeeAwLh9Z_plant_31_20240504195537225305"
        mapName = ""
        split = "train"
        mapPath = f"./out/maps/multiPointTrajs/learner_{split}/{mapName}/"
        predPath = f"./out/RoboHop/learner_{split}/{mapName}/"
    else:
        mapPath = sys.argv[1]
        predPath = sys.argv[2]
    main(mapPath, predPath, modelResumePath, split)


# %%
# Run example
# python learner.py ./out/maps/multiPointTrajs/5cdEh9F2hJL_toilet_36_20240430160015093231/ ./out/runs/dump/5cdEh9F2hJL_toilet_36_20240430160015093231_20240430164947798366/controlLogs/
# python learner.py ./out/maps/multiPointTrajs/learner_v01/h1zeeAwLh9Z_plant_31_20240504195537225305/ ./out/RoboHop/learner_v01/h1zeeAwLh9Z_plant_31_20240504195537225305/

# %%
# # VIS TEST SNIPPET
split = 'val'
mapPath = f"./out/maps/multiPointTrajs/learner_{split}/"
predPath = f"./out/RoboHop/learner_{split}/"
mapName = natsorted(os.listdir(mapPath))[0]

vecs, actions, [masks,coords] = loadData_mapOnly(mapPath+mapName,predPath+mapName)
# dataset = TrajVectorDataset(vecs, actions)
imgIdx = 15
masks, coords = masks[imgIdx], coords[imgIdx]
weights = vecs[imgIdx][:,2]
# predict weights from the model
model = controlModel(input_size=N_I, hidden_size=N_H, outdim=OUTDIM,format=FORMAT,model_type=MODEL_TYPE).cuda()
model.load_state_dict(torch.load('./out/learner/trained_models/_20240522131628952319/latest.pth'))
model.eval()
weights = model(torch.tensor(vecs[imgIdx][None],dtype=torch.float32).cuda())[1].detach().cpu().numpy().flatten()
vis = drawWeightedSegmentation([f'{mapPath}/{mapName}',f'{predPath}/{mapName}'],imgIdx,weights)
plt.imshow(vis)

# %%
mapName = ""
split = 'val'
mapPath = f"./out/maps/multiPointTrajs/learner_{split}/{mapName}/"
predPath = f"./out/RoboHop/learner_{split}/{mapName}/"
vecs, labs = loadData_mapOnly_multi(mapPath, predPath, precomDataPath=f'./out/learner/{split}.pkl')

dataloader = get_dataloader(vecs, labs, batch_size=1)
model = controlModel(input_size=N_I, hidden_size=N_H, outdim=OUTDIM, format=FORMAT,model_type=MODEL_TYPE).cuda()
modelResumePath = "./out/learner/trained_models/_20240524230801940696/latest.pth"
model.load_state_dict(torch.load(modelResumePath)) if modelResumePath else None
preds, labs = test(model, dataloader)
# plot a histogram of the errors
errors = np.abs(preds - labs)
plt.hist(errors, bins=100, density=True)
plt.show()
# compute errs for labels > 0, = 0, < 0, separately
errs = np.abs(np.array(preds) - np.array(labs))
errs_pos = errs[labs > 0]
errs_neg = errs[labs < 0]
errs_zero = errs[labs == 0]
print(f'Mean error: {errs.mean()}')
print(f'Mean error for positive labels: {errs_pos.mean()}')
print(f'Mean error for negative labels: {errs_neg.mean()}')
print(f'Mean error for zero labels: {errs_zero.mean()}')


# %%
