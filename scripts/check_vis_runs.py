# %%

import os
import h5py
import pickle
import numpy as np
from natsort import natsorted
import sys
import networkx as nx
from os.path import expanduser
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0,"../auto_agents/SegmentMap/")
sys.path.insert(0,"../")
import func
import auto_agent as AA

logFiles = {"logs.txt", "logs2.txt", "logs3.txt", "logs4.txt"}
outdir = "../out/RoboHop/go_stanford/"
imgDir = f"{expanduser('~')}/fastdata/navigation/go_stanford/"

def get_pl_per_img(G,goalNodeIdx,imgIdx):
    if type(G) != nx.classes.graph.Graph:
        G_ = G.graph
        goalNodeIdx = G.get_contracted_node(goalNodeIdx)
        nodeInds = [G.get_contracted_node(n) for n in G_.nodes() if G_.nodes[n]['map'][0] == imgIdx]
    else:
        G_ = G
        nodeInds = [n for n in G_.nodes() if G_.nodes[n]['map'][0] == imgIdx]
    pls = [nx.shortest_path_length(G_, n, goalNodeIdx ) for n in nodeInds]
    return np.array(pls)

def visualize_pl_per_img(G,imgIdx=0):
    if type(G) == nx.classes.graph.Graph:
        nodeID_to_imgRegionIdx = np.array([G.nodes[node]['map'] for node in G.nodes()])
        goalNodeIdx = len(G.nodes)-1
    else:
        nodeID_to_imgRegionIdx = np.array([G.graph.nodes[node]['map'] for node in G.graph.nodes()])
        goalNodeIdx = G.get_contracted_node(len(G.graph.nodes)-1)
    imgDir = "/home/sourav/fastdata/navigation/go_stanford/no1vcF_7_0/"
    pl_0 = get_pl_per_img(G,goalNodeIdx,imgIdx)
    img = cv2.imread(f"{imgDir}/{imgIdx}.jpg")
    nodeInds = np.argwhere(nodeID_to_imgRegionIdx[:,0] == imgIdx).flatten()
    if type(G) == nx.classes.graph.Graph:
        masks = np.array([func.rle_to_mask(G.nodes[n]['segmentation']) for n in nodeInds])
    else:
        masks = np.array([func.rle_to_mask(G.graph.nodes[G.get_contracted_node(n)]['segmentation']) for n in nodeInds])
    colors, norm = AA.value2color(pl_0,cmName='jet')
    vis = func.drawMasksWithColors(img,masks,colors)
    plt.imshow(vis)
    plt.colorbar()

def get_nodeID_to_imgRegionIdx(G):
    if type(G) == nx.classes.graph.Graph:
        nodeID_to_imgRegionIdx = np.array([G.nodes[node]['map'] for node in G.nodes()])
    else:
        nodeID_to_imgRegionIdx = np.array([G.graph.nodes[node]['map'] for node in G.graph.nodes()])
    return nodeID_to_imgRegionIdx

def change_edge_attr(G):
    for e in G.edges(data=True): 
        if 'margin' in e[2]:
            e[2]['margin'] = 0.0
    return G

def getPaths(G,nodeID_to_imgRegionIdx,imgIdx,goalNodeIdx=None,weight=None):
    if goalNodeIdx is None:
        goalNodeIdx = len(G.nodes)-1
    nodeInds = np.argwhere(nodeID_to_imgRegionIdx[:,0] == imgIdx).flatten()
    pls = [nx.shortest_path_length(G,n,goalNodeIdx,weight=weight) for n in nodeInds]
    masks = np.array([func.rle_to_mask(G.nodes[n]['segmentation']) for n in nodeInds])
    return nodeInds, masks, pls

def createVisVideo(G4,n2ir4,imgdir,outdir,subdir,imgW=160,imgH=120):
    # init video writer
    outVidPath = f"{outdir}/{subdir}/pl_vis_contracted.mp4"
    # skip if video already exists
    if os.path.exists(outVidPath):
        print(f"\t \t Skipping {subdir}, video already exists")
        return
    else:
        out = cv2.VideoWriter(outVidPath,cv2.VideoWriter_fourcc(*'mp4v'), 1, (imgW,imgH), True)

    for i,imgIdx in enumerate(range(n2ir4[-1,0])):
        nodeInds, masks, pls = getPaths(G4,n2ir4,imgIdx,weight='margin')
        colors, norm = AA.value2color(pls,cmName='jet')
        assert(keys[i] == f"{imgIdx}.jpg")
        img = cv2.imread(f"{imgdir}/{subdir}/{imgIdx}.jpg")
        vis = func.drawMasksWithColors(img,masks,colors)
        out.write((vis*255).astype(np.uint8)[:,:,::-1])
    out.release()

inds2recompute = []
inds2recompute_noh5 = []
# iterate over all subdirs
for di, subdir in enumerate(tqdm(natsorted(os.listdir(imgDir)))):
    print(f"Subdir: {di} {subdir}")
    # check if they have nodes.h5, if they have nodes_graphObject_4.pkl, and how many images in nodes.h5
    h5FullPath = f"{outdir}/{subdir}/nodes.h5"
    graphPath_3 = f"{outdir}/{subdir}/nodes_graphObject_3.pickle"
    graphPath_4 = f"{outdir}/{subdir}/nodes_graphObject_4.pickle"
    if not os.path.exists(h5FullPath):
        print(f"\t Skipping {subdir}, no h5 file found")
        inds2recompute_noh5.append(di)
        continue
    try:
        with h5py.File(h5FullPath, "r") as f:
            keys = natsorted(f.keys())
            if len(keys) == 0:
                print(f"\t Skipping {subdir}, h5 file has 0 images")
                continue
            else:
                print(f"\t \t Number of images: {len(keys)}")
    except:
        print(f"\t Skipping {subdir} due to following error: {sys.exc_info()}")
        continue

    if not os.path.exists(graphPath_4):
        print(f"\t \t Skipping {subdir}, no pkl file found")
        continue
    else:
        try:
            G4 = pickle.load(open(graphPath_4, "rb"))
        except:
            print(f"\t \t Skipping {subdir} due to following error: {sys.exc_info()}")
            continue
        if not nx.is_connected(G4):
            print(f"\t \t Skipping {subdir}, graph is not connected")
            # print size of connected components
            print("Subgraphs:", [len(c) for c in nx.connected_components(G4)])
            # print subgraph's (first few) nodes
            for si,c in enumerate(nx.connected_components(G4)):
                if si == 0: continue
                print(f"Num nodes {len(c)}: ", list(c)[:5])
                print(f"Num edges {len(G4.subgraph(c).edges)}")
                print(G4.subgraph(c).nodes(data=True))
            inds2recompute.append(di)
            continue
            # G4 = func.modify_graph(G4, G4.nodes(data=True), G4.edges(data=True)+G4.graph['temporalEdges'])
        G4 = change_edge_attr(G4)
        n2ir4 = get_nodeID_to_imgRegionIdx(G4)
        # createVisVideo(G4,n2ir4,imgDir,outdir,subdir,imgW=160,imgH=120)
np.save("../out/inds2recompute.npy", inds2recompute)
np.save("../out/inds2recompute_noh5.npy", inds2recompute_noh5)

# %%
