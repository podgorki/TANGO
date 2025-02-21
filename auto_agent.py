import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
import networkx as nx
from natsort import natsorted
import pickle
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import Delaunay

import torch
from PIL import Image as PILImage
import cv2


# NOTE: cd <drive_any_robot>; pip install -e train --no-dependencies
# from auto_agents.gnm.utils import to_numpy, transform_images, load_model

sys.path.insert(0,"./auto_agents/SegmentMap/")
from auto_agents.SegmentMap import func, zmq_server_socket
from auto_agents.SegmentMap.LightGlue.lightglue.utils import load_image, rbd, resize_image, numpy_image_to_torch

class Agent_RoboHop():
    def __init__(self,imgDir,modelsPath="./models/",h5FullPath="./out/RoboHop/nodes.h5",da_sim_thresh=0.9,da_global=True,forceRecomputeGraph=True,device=torch.device('cuda'),args=None,**kwargs):
        """
        imgDir: path to dir containing images (from the mapping run)
        modelsPath: path where model files are stored such as sam_vit_h_4b8939.pth
        h5FullPath: fullpath to h5 file to store precomputed data from processed images
        da_sim_thresh: similarlity threshold for data association (DA) edges
        da_global: whether to match images globally or locally (i.e., current to next)
        forceRecomputeGraph: whether to recompute the graph or load from precomputed file
        device: device to run the models on
        """
        self.args = args
        self.device = device
        self.h5FullPath = h5FullPath
        self.imgDir = imgDir
        self.cfg = { "desired_width": 320, "desired_height": 240, "detect": 'dinov2', "use_sam": True, "class_threshold": 0.9, \
       "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
              "dino_strides": 4, "use_traced_model": False,
              "img_sidx": 0, "img_eidx": None, "img_step": 1,
              "rmin":0, "DAStoreFull":False, "wrap":False} # robohop specifc params
        self.cfg.update(kwargs)
        clipCSV = self.args.clip_csv
        # indirectly disable clip based filtering (cosine dis always < 2.0) 
        if clipCSV == "":
            self.ftRemTh = 2.0
            self.cfg['skipCLIP'] = True
        else:
            self.ftRemTh = 0.23
            self.cfg['skipCLIP'] = False

        self.img_h = self.cfg['desired_height']
        self.img_w = self.cfg['desired_width']
        sam_checkpoint = f"{modelsPath}/sam_vit_h_4b8939.pth"

        print("Loading models...")
        self.models = func.loadAllModels(sam_checkpoint, self.cfg)
        # use local feature matching
        self.lexor, self.lmatcher = func.loadDAModels()

        self.ims = natsorted(os.listdir(f'{self.imgDir}'))
        ims_sidx, ims_eidx, ims_step = self.cfg['img_sidx'], self.cfg['img_eidx'], self.cfg['img_step']
        self.ims = self.ims[ims_sidx:ims_eidx][::ims_step]
        print(f"{len(self.ims)} images in the map directory {self.imgDir}")
        if os.path.exists(self.h5FullPath) and not self.cfg['forceRecomputeH5']:
            print(f"Using precomputed h5 file {self.h5FullPath}...")
        else:
            print(f"Processing images and saving to h5 file {self.h5FullPath}...")
            os.makedirs(os.path.dirname(self.h5FullPath), exist_ok=True)
            func.process_images_to_h5(self.h5FullPath,self.cfg,self.ims,self.models,dataDir=self.imgDir)

        if self.cfg['skipCLIP']:
            self.txt_ft = None
        else:
            self.txt_ft = func.getCLIPtxtFeat(clipCSV,self.models[3],self.models[2]).detach().cpu().numpy()

        gPath = f'{self.h5FullPath[:-3]}_graphObject.pickle'
        if os.path.exists(gPath) and not forceRecomputeGraph:
            self.G = pickle.load(open(gPath,'rb'))
        else:
            print("Creating graph...")
            if self.cfg['detect'] is not None:
                self.G = func.create_env_graph(self.h5FullPath,self.ims, dataDir=self.imgDir,
                    # ft2remove=nodeInds2remove_rftDA, ftType='da', ftRemTh=0.5,
                    ft2remove=self.txt_ft, ftType='txt', ftRemTh=self.ftRemTh,
                    minSegArea=400, aggFtNbrs=False, temporalEdge=True,
                    intraNbrsAll=kwargs['intraNbrsAll'],
                    wrapEdgesPano=False
                    )
            else:
                self.G = func.create_env_graph_simple(self.h5FullPath,self.ims, dataDir=self.imgDir,intraNbrsAll=kwargs['intraNbrsAll'])
                # pickle.dump(self.G,open(gPath,'wb'))

        self.rft_da_env_arr = self.G.graph['rft_da_env_arr']
        rft_lang_env_arr = self.G.graph['rft_lang_env_arr']
        self.nodeID_to_imgRegionIdx = np.array([self.G.nodes[node]['map'] for node in self.G.nodes()])
        
        if self.args.use_depth:
            if not self.G.graph.get('has_depth',False):
                print("Loading depth data into G...")
                self.depthDir = imgDir.replace('images','images_depth')
                for i,im in enumerate(tqdm(self.ims)):
                    im = im.replace('.png','.npy')
                    dep = np.load(f'{self.depthDir}/{im}',allow_pickle=True)
                    dep = self.preProcDepth(dep)
                    nodeIndsPerImg = np.argwhere(self.nodeID_to_imgRegionIdx[:,0] == i).flatten()
                    for j,nodeIdx in enumerate(nodeIndsPerImg):
                        self.G.nodes[nodeIdx]['depth'] = self.getMeanDepthForNode(self.G.nodes[nodeIdx]['segmentation'],dep)
                self.G.graph['has_depth'] = True
                pickle.dump(self.G,open(gPath,'wb'))

        self.compute_graphs(da_sim_thresh,da_global,forceRecomputeGraph,comp_G4=kwargs.get('comp_G4',True),comp_G23=kwargs.get('comp_G23',False))

        print("Precomputing DA nbr list...")
        self.daNbrs = self.precompute_nbrs(self.G4,edgeType='da')

        print("Precomputing path lengths between all node pairs...")
        self.mapNodeWeightStr = self.args.weight_string
        self.allPathLengths = self.get_path(None,None,self.G4,self.mapNodeWeightStr,allPairs=True)

        self.currImgInfo = None

        # setup controller
        self.map_nodes_coords = np.array(([np.array(np.nonzero(self.G.nodes[n]['segmentation'])).mean(1)[::-1].astype(int) for n in self.G.nodes]))
        self.multiRep = False
        self.controller = zmq_server_socket.RoboHopModelAndServer(None,None,agent=self,logsPath=kwargs.get('controlLogsDir',None)) # FIX: self arg
        self.liveImgIter = 0
        self.localizedImgIdx = 0
        self.goalNode = 0
        self.goalNodeNbrs = None
        self.pathLengths = []
        self.trackFail = False
        self.done = False
        self.hop = False
        self.reloc_rad = 5
        self.localizer_iter_lb = 0
        self.gain_lin = 0.01
        self.gain_ang = 0.2
        self.control_mode = 'CC'

        self.followFloor = False

        self.learnersDict = []
        if self.args.controlModelPath is not None:
            print(f"Loading trained control model from path {self.args.controlModelPath}...")
            from train.model import controlModel
            FORMAT, OUTDIM = 'regBoth', 2
            MODEL_TYPE = 'mlp2'
            N_H, N_I = 32, 100
            self.controlModel = controlModel(input_size=N_I, hidden_size=N_H, outdim=OUTDIM,format=FORMAT,model_type=MODEL_TYPE).cuda()
            self.controlModel.load_state_dict(torch.load(self.args.controlModelPath))
            self.controlModel.eval()

    def getImg(self,imgIdx):
        return func.loadPreProcImg(f'{self.imgDir}/{self.ims[imgIdx]}',self.cfg)

    def getNodeInds(self,imgIdx):
        return np.argwhere(self.nodeID_to_imgRegionIdx[:,0] == imgIdx).flatten()
        
    def preProcDepth(self,dep):
        return cv2.resize(dep[self.cfg['rmin']:], (self.cfg['desired_width'],self.cfg['desired_height']),interpolation=cv2.INTER_NEAREST)

    def getMeanDepthForNode(self,mask_seg,dep):
        mask_depNonZero = dep!=0 # habitat has some 0 depth valued regions
        depVals = dep[mask_seg*mask_depNonZero]
        return depVals.mean() if depVals.size > 0 else 1e6 # set a max value for 0 depth regions

    def compute_graphs(self,da_sim_thresh=0.9,da_global=True,forceRecomputeGraph=False,comp_G4=True,comp_G23=True):
        self.G2, self.G3, self.G3_, self.G4 = None, None, None, None 

        intraImage_edges, da_edges, temporal_edges = func.getSplitEdgeLists(self.G,flipSim=True)

        if self.args.use_depth4map:
            g3dPath = f'{self.h5FullPath[:-3]}_graphObject_3d.pickle'
            if os.path.exists(g3dPath) and not forceRecomputeGraph:
                self.G = pickle.load(open(g3dPath,'rb'))
            else:
                print("Computing 3D edges for G...")
                nbrsAll = []
                for i,im in enumerate(tqdm(self.ims)):
                    nodeIndsPerImg = np.argwhere(self.nodeID_to_imgRegionIdx[:,0] == i).flatten()
                    nodeDepth = self.nodes2key(nodeIndsPerImg,'depth')
                    nodeCoords = self.nodes2key(nodeIndsPerImg,'coords')
                    nodeCoords3d = np.concatenate((nodeCoords,nodeDepth[:,None]),axis=1)
                    nbrs = create_edges_DT(nodeCoords3d)
                    nbrs = nodeIndsPerImg[nbrs]
                    nbrsAll.append(nbrs)
                intraImage_edges = np.vstack(nbrsAll).tolist()
                self.G = func.modify_graph(self.G, self.G.nodes(data=True), da_edges + temporal_edges + intraImage_edges)
                pickle.dump(self.G,open(g3dPath,'wb'))

        if comp_G4:
            print("\nConverting Graph G to G4...")
            g4Path = f'{self.h5FullPath[:-3]}_graphObject_4.pickle'
            if os.path.exists(g4Path) and not forceRecomputeGraph:
                self.G4 = pickle.load(open(g4Path,'rb'))
            else:
                da_edges_rob, temporal_edges_rob = self.get_robust_DA_edges(win=3)
                print(f"Num robust DA edges: {len(da_edges_rob)}")
                self.G4 = func.modify_graph(self.G, self.G.nodes(data=True), da_edges_rob+intraImage_edges)
                pickle.dump(self.G4,open(g4Path,'wb'))
            print(f"Num Nodes and eges: {self.G4.number_of_nodes()}, {self.G4.number_of_edges()}")

        if comp_G23:
            print("\nComputing DA edges from dMat...")
            self.dMat, sims, matches = func.getRegDisMat(self.G,numAggLayers=0,show_dmat=False,p=1)
            wrap = len(self.ims) if self.cfg['wrap'] else None
            da_edges = func.reProcDAEdges(self.dMat,self.nodeID_to_imgRegionIdx,simThresh=da_sim_thresh,wrap=wrap,globalSearch=da_global)
            print(f"Num DA edges: {len(da_edges)}")

            print("\nConverting Graph G to create G2...")
            g2Path = f'{self.h5FullPath[:-3]}_graphObject_2.pickle'
            if os.path.exists(g2Path) and not forceRecomputeGraph:
                self.G2 = pickle.load(open(g2Path,'rb'))
            else:
                self.G2 = func.modify_graph(self.G,self.G.nodes(data=True),intraImage_edges+da_edges+temporal_edges)
                pickle.dump(self.G2,open(g2Path,'wb'))
        else:
            self.G2 = self.G4

        if self.cfg.get('comp_G3',True):
            print("\nConverting Graph G2 to create G3...")
            g3Path = f'{self.h5FullPath[:-3]}_graphObject_3.pickle'
            if os.path.exists(g3Path) and not forceRecomputeGraph:
                self.G3 = pickle.load(open(g3Path,'rb'))
            else:
                self.G3, nodes_removed, nodes_retained = func.contractNodes_DA(self.G2, da_edges+temporal_edges)
                pickle.dump(self.G3,open(g3Path,'wb'))
            self.G3_ = self.G3.graph
            print(f"Num Nodes and eges: {self.G3_.number_of_nodes()}, {self.G3_.number_of_edges()}")

    def precompute_nbrs(self,G,edgeType='da'):
        nbrs = []
        for node in G.nodes():
            nbrs.append([n for n in G.neighbors(node) if G.edges[node,n].get('edgeType')==edgeType])
        return nbrs

    def init_episode(self,minTrajLength=10,goalNode=None,startNode=None,seed=0,goalImgIdx=None,saveVisual=True):
        if goalNode is None:
            # pick a random image index from refMap > trajLength
            # set seed
            np.random.seed(seed)
            if goalImgIdx is None:
                goalImgIdx = np.random.randint(minTrajLength,len(self.ims))

            # pick a random node from the goalImg
            nodesInGoalImg = np.argwhere(self.nodeID_to_imgRegionIdx[:,0] == goalImgIdx).flatten()
            self.goalNode = np.random.choice(nodesInGoalImg)
        else:
            self.goalNode = goalNode
            goalImgIdx = self.nodeID_to_imgRegionIdx[goalNode][0]

        # get goal node neighbors from the same img
        goalNodeNbrs = list(self.G4.neighbors(self.goalNode))
        self.goalNodeNbrs = [n for n in goalNodeNbrs if self.nodeID_to_imgRegionIdx[n][0] == goalImgIdx]
        self.goalNodeNbrsImgIdx = self.nodeID_to_imgRegionIdx[self.goalNodeNbrs,0]

        initImgIdx = goalImgIdx - minTrajLength
        self.localizedImgIdx = initImgIdx - 1 # to avoid exact matching between nodes

        if startNode is None:
            # pick a random init node from initImg
            nodesInInitImg = np.argwhere(self.nodeID_to_imgRegionIdx[:,0] == initImgIdx).flatten()
            startNode = np.random.choice(nodesInInitImg)
  
        path = self.get_path(startNode,self.goalNode,self.G4,'margin')
        self.plan = path
        self.episode = {"seed":seed, "goalNode":self.goalNode, "startNode":startNode,
                        "goalImgIdx":goalImgIdx, "initImgIdx":initImgIdx,
                        "goalNodeNbrs":self.goalNodeNbrs,
                        "goalNodeNbrsImgIdx:": self.goalNodeNbrsImgIdx, "plan_init":path, "localizedImgIdx":self.localizedImgIdx, "minTrajLength":minTrajLength,"args":self.args}
        # generate visuals
        if saveVisual:
            startNodeImgVis = func.showSegFromH5(self.h5FullPath,self.ims,initImgIdx,self.nodeID_to_imgRegionIdx[startNode,1],cfg=self.cfg,dataDir=self.imgDir,doPlot=False,drawBG=False)
            goalNodeImgVis = func.showSegFromH5(self.h5FullPath,self.ims,goalImgIdx,self.nodeID_to_imgRegionIdx[self.goalNode,1],cfg=self.cfg,dataDir=self.imgDir,doPlot=False,drawBG=False)
            goalNodeNbrsImgVis = func.showSegFromH5(self.h5FullPath,self.ims,goalImgIdx,self.nodeID_to_imgRegionIdx[self.goalNodeNbrs,1],cfg=self.cfg,dataDir=self.imgDir,doPlot=False,drawBG=False)
            combinedImg = np.concatenate([startNodeImgVis,goalNodeImgVis,goalNodeNbrsImgVis],axis=1)
            # single call putText start node, goal node, init img, goal img indices
            txtString = f"Start Node: {startNode}, Goal Node: {self.goalNode}, Init Img: {initImgIdx}, Goal Img: {goalImgIdx}"
            cv2.putText(combinedImg, txtString, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            # write all txt together
            cv2.imwrite(f"{self.controller.folder_path}/episode.png", combinedImg[:,:,::-1]*255)

    def save_episode(self,**kwargs):
        self.episode.update(kwargs)
        self.episode['liveImgIter'] = self.liveImgIter
        self.episode['plan_executed'] = self.plan
        self.episode['hops'] = self.controller.iter
        self.episode['pathLengths'] = np.array(self.pathLengths)
        self.episode['localizedImgIdx'] = self.localizedImgIdx
        self.episode['learnersDict'] = self.learnersDict
        np.save(f'{self.controller.folder_path}/episode.npy', self.episode)

    def nodes2key(self,nodeInds,key):
        _key = key
        if key == 'coords':
            _key = 'segmentation'
        if isinstance(nodeInds[0],dict):
            if _key == 'segmentation' and self.cfg.get('compressMask',False):
                values = np.array([func.rle_to_mask(n[_key]) for n in nodeInds])
            else:
                values = np.array([n[_key] for n in nodeInds])
        else:
            if _key == 'segmentation' and self.cfg.get('compressMask',False):
                values = np.array([func.rle_to_mask(self.G.nodes[n][_key]) for n in nodeInds])
            else:
                values = np.array([self.G.nodes[n][_key] for n in nodeInds])
        if key == 'coords':
            values = np.array([np.array(np.nonzero(v)).mean(1)[::-1].astype(int) for v in values])
        return values

    def map_node2kp(self,kp,nodeInds):
        masks, areas = self.nodes2key(nodeInds,'segmentation'), self.nodes2key(nodeInds,'area')
        m = masks[:,kp[:,1].astype(int),kp[:,0].astype(int)]
        return m, areas

    def loadImg_LFM(self,imgIdx):
        if isinstance(imgIdx,(int,np.integer)):
            img = load_image(f'{self.imgDir}/{self.ims[imgIdx]}',resize=(self.img_h, self.img_w)).to('cuda')
        else:   
            if not isinstance(imgIdx,np.ndarray):
                imgIdx = np.array(imgIdx)
            imgIdx = imgIdx[:,:,:3]
            img = numpy_image_to_torch(resize_image(imgIdx, (self.img_h, self.img_w))[0]).to('cuda')
        return img
    
    def getLocalMatching(self,imSrcIdx,imTgtIdx,nodeInds_i,nodeInds_j):
        visualize = False
        imSrc = self.loadImg_LFM(imSrcIdx)
        imTgt = self.loadImg_LFM(imTgtIdx)
        ftSrc = self.lexor.extract(imSrc)
        ftTgt = self.lexor.extract(imTgt)
        lmatches = rbd(self.lmatcher({'image0': ftSrc, 'image1': ftTgt}))# remove batch dimension
        count, score = lmatches['matches'].shape[0], lmatches['scores'].mean(0).detach().cpu().numpy()[()]
        # print([count,score,ftSrc['keypoints'].shape[1],ftTgt['keypoints'].shape[1]])
        ftSrc, ftTgt = [rbd(x) for x in [ftSrc, ftTgt]]  # remove batch dimension
        kp1, kp2, matches = ftSrc['keypoints'], ftTgt['keypoints'], lmatches['matches']
        mkp1, mkp2 = kp1[matches[..., 0]].detach().cpu().numpy(), kp2[matches[..., 1]].detach().cpu().numpy()

        node2kp1, areas1 = self.map_node2kp(mkp1,nodeInds_i)
        node2kp2, areas2 = self.map_node2kp(mkp2,nodeInds_j)
        areaDiff_ij = areas1[:,None].astype(float) / areas2[None,:]
        areaDiff_ij[areaDiff_ij>1] = 1 / areaDiff_ij[areaDiff_ij>1]  
        lmat_ij = (node2kp1[:,None] * node2kp2[None,]).sum(-1)
        matchesBool_lfm = lmat_ij.sum(1)!=0
        lmat_ij = lmat_ij * areaDiff_ij

        matches_ij = lmat_ij.argmax(1)
        matchesBool_area = areaDiff_ij[np.arange(len(matches_ij)),matches_ij] > 0.5
        matchesBool = np.logical_and(matchesBool_lfm, matchesBool_area)

        im_lfm, im_lfm_nodes = None, None
        if visualize:
            im_lfm = np.concatenate([imSrc.cpu().numpy(),imTgt.cpu().numpy()],axis=2).transpose(1,2,0)
            im_lfm_nodes = im_lfm.copy()
            for i in range(len(mkp1)):
                cv2.line(im_lfm, (int(mkp1[i][0]), int(mkp1[i][1])),(int(mkp2[i][0] + imSrc.shape[1]), int(mkp2[i][1])), (255, 0, 0), 2, lineType=cv2.LINE_AA)
            coords_i, coords_j = self.nodes2key(nodeInds_i,'coords')[matchesBool], self.nodes2key(nodeInds_j,'coords')[matches_ij[matchesBool]] # BUG ALERT: qry should be called differently when using self.nodes2key
            for i in range(matchesBool.sum()):
                cv2.line(im_lfm_nodes, (int(coords_i[i][0]), int(coords_i[i][1])),(int(coords_j[i][0] + self.img_w), int(coords_j[i][1])), (255, 0, 0), 2, lineType=cv2.LINE_AA)

        singleBestMatch = lmat_ij.max(1).argmax()
        return matchesBool, matches_ij, singleBestMatch, lmat_ij, [im_lfm, im_lfm_nodes]

    def get_robust_DA_edges(self,topK=None,win=0):
        imgIdx_s, imgIdx_e = self.nodeID_to_imgRegionIdx[0,0],self.nodeID_to_imgRegionIdx[-1,0] + 1
        da_edges = []
        temporal_edges = []
        for i in tqdm(range(imgIdx_s, imgIdx_e)):
            nodeInds_i = np.argwhere(self.nodeID_to_imgRegionIdx[:,0] == i).flatten()
            if self.cfg['detect'] is not None:
                mat_ii = self.dMat[nodeInds_i][:,nodeInds_i]
                sortedSim_ii = np.sort(mat_ii,1)
            if win==0:
                endIdx = imgIdx_e
            else:
                endIdx = min(i+1+win,imgIdx_e)
            for j in range(i+1, endIdx):
                nodeInds_j = np.argwhere(self.nodeID_to_imgRegionIdx[:,0] == j).flatten()
                if self.cfg['detect'] is not None:
                    mat_ij = self.dMat[nodeInds_i][:,nodeInds_j]
                    # TODO: compute sortedSim_ji and avg margin both ways
                    # TODO: or mutual NN?
                    singleBestMatch, matches_ij, margin_ij, _ = find_matches(mat_ij,sortedSim_ii)
                else:
                    margin_ij = np.zeros((len(nodeInds_i)))

                # override with local matching
                matchesBool, matches_ij, singleBestMatch, _, _ = self.getLocalMatching(i,j,nodeInds_i,nodeInds_j)

                da_edges.append(np.array([nodeInds_i[matchesBool], nodeInds_j[matches_ij][matchesBool], margin_ij[matchesBool]])[:,np.argsort(-margin_ij[matchesBool])[:topK]])
                if j==i+1:
                    temporal_edges.append([nodeInds_i[singleBestMatch], nodeInds_j[matches_ij][singleBestMatch], margin_ij[singleBestMatch]])
        da_edges = np.concatenate(da_edges,axis=1).T
        temporal_edges = np.array(temporal_edges)
        da_edges = [(int(e[0]),int(e[1]),{'margin':np.exp(-100*e[2]), 'edgeType':'da'}) for e in da_edges]
        temporal_edges = [(int(e[0]),int(e[1]),{'margin':np.exp(-100*e[2]), 'edgeType':'temporal'}) for e in temporal_edges]
        return da_edges, temporal_edges


    def search_with_text(self,txtStr,clipThresh=0.25,subgraphsMinMax=[0,1],numPerSubGraph=1):
        matchedNodes_thresh, cc, sim_imgTxt = func.query_with_text(self.G2,self.models,txtStr,clipThresh,self.nodeID_to_imgRegionIdx)
        if len(cc) == 0:
            cc = [matchedNodes_thresh]
        for i,subgraph in enumerate(cc):
            if i < subgraphsMinMax[0] or i > subgraphsMinMax[1]:
                continue
            print("\nSubGraph ", i)
            for ii,elem in enumerate(subgraph):
                if ii >= numPerSubGraph: break
                imgIdx, regIdx = self.nodeID_to_imgRegionIdx[elem]
                print(f"Contracted node: {elem, self.G3.get_contracted_node(elem)}")
                print(elem,[imgIdx,regIdx],sim_imgTxt[ii])
                func.showSegFromH5(self.h5FullPath,self.ims,imgIdx,regIdx,cfg=self.cfg,dataDir=self.imgDir)

    def get_path(self, src, tgt, G=None, weight=None, allPairs=False):
        if G is None:
            if self.G4 is not None:
                G = self.G4
            elif self.G3_ is not None:
                G = self.G3_
                src = self.G3.get_contracted_node(src)
                tgt = self.G3.get_contracted_node(tgt)
            else:
                G = self.G
        if allPairs:
            # this returns lengths
            pathLengths = dict(nx.all_pairs_dijkstra_path_length(G,weight=weight))
            pathLengths = np.array([[pathLengths[src][tgt] for tgt in G.nodes()] for src in G.nodes()])
            return pathLengths
        else:
            # this returns paths
            shortest_path = nx.shortest_path(G,source=src,target=tgt,weight=weight)
            return shortest_path

    def display_path_images(self, path, display=True):
        pathSegmentImages = []
        for ii,elem in enumerate(path):
            # if not (0 <= ii < 5): continue
            # if ii!=len(shortest_path)-1:
                # print(G2.edges[elem,shortest_path[ii+1]])
            #     print(f"Original edges",H.get_original_edges(elem,shortest_path[ii+1]))
                # print(f"Orig2", [(n1,n2) for n1 in G3.get_constituents(elem) for n2 in G3.get_constituents(shortest_path[ii+1]) if (n1,n2) in G2.edges()])
            # print(G3.graph.nodes[elem]['area'])
            imgIdx, regIdx = self.nodeID_to_imgRegionIdx[elem]
            print(f"Node {elem} in image {imgIdx} at region {regIdx}")
            # print(f"Constituents of node {elem}: {G3.get_constituents(elem)}")

            pathSegmentImg = func.showSegFromH5(self.h5FullPath,self.ims,imgIdx,regIdx,cfg=self.cfg,dataDir=self.imgDir,doPlot=display)
            pathSegmentImages.append((pathSegmentImg*255).astype(np.uint8))
        return pathSegmentImages

    def getSubMap(self,currImgIndexInRef,searchRad=5):
        # relevant images to search for segments (~local map / ~topK VPR)
        k = searchRad//2
        refImageInds = np.concatenate([np.arange(currImgIndexInRef-k, currImgIndexInRef),np.arange(currImgIndexInRef+1, currImgIndexInRef+k+1)])
        nodeIndsFromRef = np.concatenate([np.argwhere(self.nodeID_to_imgRegionIdx[:,0] == refInd).flatten() for refInd in refImageInds]).flatten()
        rft_dino_ref = self.rft_da_env_arr [nodeIndsFromRef]
        print(f"rft_dino_ref.shape (numNodes x ftDim): {rft_dino_ref.shape}")
        subMapInfo = {
            'nodeIndsFromRef':nodeIndsFromRef,
            'rft_dino_ref':rft_dino_ref,
            }
        return subMapInfo
        
    def process_currImg(self,currImg=5,filtering=True):
        # self, currImg = RH, 5
        if isinstance(currImg,int):
            # assume a random image index from refMap is given
            currImgIndexInRef = currImg
            currImg = func.loadPreProcImg(self.imgDir+self.ims[currImg],self.cfg)
        elif isinstance(currImg,str) or isinstance(currImg,np.ndarray):
            currImg = func.loadPreProcImg(currImg,self.cfg)
        else:
            raise ValueError("currImg must be int, str, or np.ndarray")

        im_p, ift_dino, ift_clip, rft_dino, rft_clip, nodesCurr = func.process_single(self.cfg,currImg[self.cfg['rmin']:],self.models)
        rft_clip  = rft_clip.detach().cpu().numpy()
        if filtering:
            keepInds = func.filterNodes(rft_clip,nodesCurr,self.txt_ft,minSegArea=400,ftRemTh=self.ftRemTh)
        else:
            keepInds = np.arange(len(masks))
        nodesCurr = np.array([nodesCurr[k] for k in keepInds])
        if self.args.use_depth:
            for nix,node in enumerate(nodesCurr):
                nodesCurr[nix]['depth'] = self.getMeanDepthForNode(node['segmentation'],self.currDepth)
        rft_clip = rft_clip[keepInds]
        rft_dino  = rft_dino.detach().cpu().numpy()[keepInds]
        rft_dino /= np.linalg.norm(rft_dino,axis=1)[:,None]
        # rft_dino = func.normalizeFeat(rft_dino)
        currImg_mask_coords = getCoordsFromNodes(nodes=nodesCurr)
        currImg_mask_area = np.array([m['area'] for m in nodesCurr])
        masks = np.array([m['segmentation'] for m in nodesCurr])

        # add all currImg info to dict
        self.currImgInfo = {
            'currImg':currImg, 'currImg_mask_coords':currImg_mask_coords, 'currImg_mask_area':currImg_mask_area, 'rft_dino':rft_dino, 
            'nodesCurr':nodesCurr, 
            'ift_dino':ift_dino, 
            'ift_clip':ift_clip, 
            'rft_clip':rft_clip, 
            'im_p':im_p,
            'masks': masks,
            }

    def procCurr(self,currImgIndexInRef=5,nodeIdx=0,imgpath=None):
        self.process_currImg(currImgIndexInRef)
        currImg_mask_coords, currImg_mask_area, currImg_rft_dino = self.currImgInfo['currImg_mask_coords'], self.currImgInfo['currImg_mask_area'], self.currImgInfo['rft_dino']
        currImg, masks, nodesCurr = self.currImgInfo['currImg'], self.currImgInfo['masks'], self.currImgInfo['nodesCurr']
        curr_node_desc = currImg_rft_dino.copy()
        print(f"curr_node_desc.shape: {curr_node_desc.shape}")
        
        refNodeIter = 0
        confs = []
        exhausted = False
        while 1:
            self.trackFail = False
            # assume localization
            givenNodeFromPlan = self.plan[nodeIdx:nodeIdx+2]

            lastNode = False
            if len(givenNodeFromPlan) == 1:
                lastNode = True
                givenNodeFromPlan = [self.plan[-1],self.plan[-1]]

            if self.multiRep and not lastNode:
                givenNodeFromPlan_multi = [list(self.G3.get_constituents(n)) for n in givenNodeFromPlan]
                numConsti = [len(constiList) for constiList in givenNodeFromPlan_multi]
                givenNodeFromPlan_multi = np.concatenate(givenNodeFromPlan_multi)
                match_currNodes_to_giveNode_multi = curr_node_desc @ self.rft_da_env_arr[givenNodeFromPlan_multi].T
                givenNode_in_currImg_multi = np.argmax(match_currNodes_to_giveNode_multi,0)
                conf_multi = match_currNodes_to_giveNode_multi[givenNode_in_currImg_multi,range(len(givenNodeFromPlan_multi))]
                bestInds_multi = np.array([conf_multi[:numConsti[0]].argmax(), numConsti[0] + conf_multi[numConsti[0]:].argmax()])
                conf = conf_multi[bestInds_multi]
                givenNodeFromPlan = givenNodeFromPlan_multi[bestInds_multi]
                givenNode_in_currImg = givenNode_in_currImg_multi[bestInds_multi]
            else:
                match_currNodes_to_giveNode = curr_node_desc @ self.rft_da_env_arr[givenNodeFromPlan].T
                givenNode_in_currImg = np.argmax(match_currNodes_to_giveNode,0)

                # override matches with LFM
                givenNode_in_currImg = []
                # LFM plan ref node img w currImg; find ref node's best match based on local match count 
                for n_i, n in enumerate(givenNodeFromPlan):
                    n_imgIdx = self.nodeID_to_imgRegionIdx[n][0]
                    n_imgIdx_nodes = np.argwhere(self.nodeID_to_imgRegionIdx[:,0] == n_imgIdx).flatten()
                    n_idxInNodes = n_imgIdx_nodes.tolist().index(n)
                    matchesBool , matches_ij, singleBestMatch, lmat_ij, im_lfm = self.getLocalMatching(n_imgIdx, currImgIndexInRef,n_imgIdx_nodes,nodesCurr)
                    lmat_ij_n = lmat_ij[n_idxInNodes]
                    sortedInds = (-lmat_ij_n).flatten().argsort()
                    idx = sortedInds[0]
                    currConf = match_currNodes_to_giveNode[idx,n_i]
                    confs.append(currConf)
                    areaRatio_map2qry = float(self.G.nodes[n]['area']) / float(nodesCurr[idx]['area'])
                    crit_area = (areaRatio_map2qry > (1/(1+2*self.controller.areaThresh)) and areaRatio_map2qry < (1+2*self.controller.areaThresh))
                    if (not crit_area or currConf < self.controller.simThresh) and n_i==0 and not exhausted:
                        print(f"Track Fail: Conf={currConf}, AreaRatio={areaRatio_map2qry}")
                        self.trackFail = True
                        break
                    givenNode_in_currImg.append(idx)
                if self.trackFail:
                    if refNodeIter == 0:
                        refNodes_plSorted = self.relocalize_replan(currImgIndexInRef,curr_node_desc,nodesCurr)[0]
                        confs = []
                    if refNodeIter == len(refNodes_plSorted):
                        print("\nExhausted relocalization node search")
                        exhausted = True
                        maxConf = np.max(confs)
                        if maxConf < self.controller.simThresh:
                            self.reloc_rad *= 2
                            print(f"MaxConf: {maxConf} still < SimThresh: {self.controller.simThresh}. Searching within radius {self.reloc_rad} ...")
                            # relocalize broader
                            refNodes_plSorted = self.relocalize_replan(currImgIndexInRef,curr_node_desc,nodesCurr)[0]
                            exhausted = False
                            refNode = refNodes_plSorted[0]
                            refNodeIter = 1
                            confs = []
                        else:
                            refNode = refNodes_plSorted[np.argmax(confs)]
                    else:
                        refNode = refNodes_plSorted[refNodeIter]
                        refNodeIter += 1
                    self.localizedImgIdx = self.nodeID_to_imgRegionIdx[refNode][0]
                    newPlan = self.get_path(refNode,self.goalNode,self.G4,'margin')
                    self.controller.iter += 1
                    self.plan = self.plan[:self.controller.iter] + newPlan
                    nodeIdx = self.controller.iter
                else:
                    break
        givenNode_in_currImg = np.array(givenNode_in_currImg)
        conf = match_currNodes_to_giveNode[givenNode_in_currImg,range(len(givenNodeFromPlan))]

        givenNodeFromPlan_imgIdx = self.nodeID_to_imgRegionIdx[givenNodeFromPlan,0]
        givenNodeFromPlan_regIdx = self.nodeID_to_imgRegionIdx[givenNodeFromPlan,1]
        givenNodeFromPlan_coord = self.map_nodes_coords[givenNodeFromPlan]
        givenNodeFromPlan_area = [self.G.nodes[n]['area'] for n in givenNodeFromPlan]
        print(f"givenNodeFromPlan_coord: {givenNodeFromPlan_coord}")
        mapNodeDict = {"nodeID": givenNodeFromPlan, "imgIdx": givenNodeFromPlan_imgIdx, "regIdx": givenNodeFromPlan_regIdx, \
                        "coord": givenNodeFromPlan_coord, "area": givenNodeFromPlan_area,}

        currLocalizeNode_coord = currImg_mask_coords[givenNode_in_currImg]
        currLocalizeNode_area = [nodesCurr[n]['area'] for n in givenNode_in_currImg]
        print(f"Nodes {givenNodeFromPlan} from Map at {givenNodeFromPlan_coord} matched to Qry Nodes {givenNode_in_currImg}\
        in curr at (r,c): {currLocalizeNode_coord}, Conf: {conf}")
        horizOffset = givenNodeFromPlan_coord[0] - currLocalizeNode_coord[0]
        # print(f"Horizontal offset: {horizOffset}")
        qryNodeDict = {"nodeID": givenNode_in_currImg, "coord": currLocalizeNode_coord, "area": currLocalizeNode_area, \
                    "conf": conf, "curr_node_desc":curr_node_desc, 'currImgInfo':self.currImgInfo}
        combinedImg = self.create_visuals(givenNodeFromPlan,confs,givenNode_in_currImg,refNodeIter>0)
        return mapNodeDict, qryNodeDict, combinedImg, self.plan

    def create_visuals(self,refNodeInds,confs,givenNode_in_currImg,relocFlag=False):
        givenNodeFromPlan = refNodeInds
        givenNodeFromPlan_imgIdx = self.nodeID_to_imgRegionIdx[givenNodeFromPlan,0]
        givenNodeFromPlan_regIdx = self.nodeID_to_imgRegionIdx[givenNodeFromPlan,1]
        # show the givenNode
        mapImg2vis = func.showSegFromH5(self.h5FullPath,self.ims,givenNodeFromPlan_imgIdx[0],givenNodeFromPlan_regIdx[0],self.cfg,doPlot=False,dataDir=self.imgDir,drawBG=False)
        mapImg2vis2 = func.showSegFromH5(self.h5FullPath,self.ims,givenNodeFromPlan_imgIdx[1],givenNodeFromPlan_regIdx[1],self.cfg,doPlot=False,dataDir=self.imgDir,drawBG=False)
        cv2.putText(mapImg2vis, f"N_m:{givenNodeFromPlan[0]} C:{100*confs[0]:.3f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(mapImg2vis, f"Db Idx: {self.localizedImgIdx}", (160, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if relocFlag:
            cv2.putText(mapImg2vis, f"Re-Localized & Re-Planned!", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if self.hop:
            cv2.putText(mapImg2vis, f"Hop!", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 255), 1)
        cv2.putText(mapImg2vis2, f"N_m:{givenNodeFromPlan[1]} C:{100*confs[1]:.3f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # show the givenNodeFromMap matched in currImg
        currImg, nodesCurr = self.currImgInfo['currImg'], self.currImgInfo['nodesCurr']
        currImg2vis, currImg2vis2 = currImg.copy(), currImg.copy()
        currImg2vis = func.get_vis_anns([nodesCurr[givenNode_in_currImg[0]]],currImg2vis.copy(),skipH5indexing=True)
        currImg2vis2 = func.get_vis_anns([nodesCurr[givenNode_in_currImg[1]]],currImg2vis2.copy(),skipH5indexing=True)
        combinedImg = np.concatenate((mapImg2vis,currImg2vis),axis=1)
        combinedImg2 = np.concatenate((mapImg2vis2,currImg2vis2),axis=1)
        combinedImg = np.concatenate((combinedImg,combinedImg2),axis=0)
        # plt.imshow(combinedImg)
        # plt.show()
        return combinedImg[:,:,::-1]

    def get_control_signal(self,idx_s,idx_e):
        for imgIdx in range(idx_s,idx_e):
            response_dict = self.controller.predict(imgIdx,imgIdx,None)

    def update_localizer_iter_lb(self):
        if self.args.greedy_propeller:
            if self.localizedImgIdx > self.localizer_iter_lb:
                self.localizer_iter_lb = self.localizedImgIdx
        else:
            self.localizer_iter_lb = max(0,self.localizedImgIdx - self.reloc_rad//2)
 
    def relocalize_replan(self,img,qry_descs,qryNodes):
        visualize = False
        removeLessReliableMatches = False
        print("\n Relocalizing (locally) and replanning...\n")
        # localize around the previous best refIdx
        k = self.reloc_rad//2
        ref_imgInds2search = np.arange(self.localizer_iter_lb,min(self.localizer_iter_lb+self.reloc_rad,len(self.ims)))
        ref_nodes2search = [i for i in range(len(self.nodeID_to_imgRegionIdx)) if self.nodeID_to_imgRegionIdx[i][0] in ref_imgInds2search]
        pls = []
        for s in ref_nodes2search:
            pl = len(self.get_path(s,self.goalNode,weight=self.mapNodeWeightStr))
            # pl = nx.shortest_path_length(self.G4,s,self.goalNode,weight='margin')
            pls.append(pl)

        # local feature matching
        matchPairs = []
        for refInd in ref_imgInds2search:
            refNodes = np.argwhere(self.nodeID_to_imgRegionIdx[:,0] == refInd).flatten()
            matchesBool , matches_ij, singleBestMatch, _, im_lfm = self.getLocalMatching(img,refInd,qryNodes,refNodes)
            if visualize: # Set to False, potential BUG, to be removed or updated
                refNodesMatched = refNodes[matches_ij[matchesBool]]
                weights = np.array([len(self.get_path(r,self.goalNode,weight=self.mapNodeWeightStr)) for r in refNodesMatched])
                colors, norm = value2color(weights)
                coords_i, coords_j = self.nodes2key(qryNodes,'coords')[matchesBool], self.nodes2key(qryNodes,'coords')[matches_ij[matchesBool]] # BUG ALERT: qry should be called differently when using self.nodes2key
                visualize_flow(coords_i,coords_j,self.currImgInfo['currImg'].copy(),colors,norm,weights)
                plt.imshow(func.drawMasksWithColors(self.currImgInfo['currImg'],self.currImgInfo['masks'][matchesBool],colors))
                plt.show()
            matchPairs.append(np.column_stack([np.argwhere(matchesBool).flatten(), refNodes[matches_ij[matchesBool]]]))
        matchPairs = np.vstack(matchPairs)
        matchedRefNodes = matchPairs[:,1]
        refNodes_plSorted = matchedRefNodes[np.argsort([pls[ref_nodes2search.index(k)] for k in matchedRefNodes])]

        if removeLessReliableMatches: # Set ot False; this section of code to be removed/updated
            # segment sim to remove less reliable matches
            refNodes_plSorted_segSim = np.max(self.rft_da_env_arr[refNodes_plSorted] @ qry_descs.T,axis=1)
            refNode = refNodes_plSorted[0]
            if refNodes_plSorted_segSim[0] < self.controller.simThresh:
                print(f"Seg Sim of best LFM node {refNode} is {refNodes_plSorted_segSim[0]} < {self.controller.simThresh}, searching ranked list for a better match...")
                for r_i, r in enumerate(refNodes_plSorted):
                    if refNodes_plSorted_segSim[r_i] > self.controller.simThresh:
                        refNode = r
                        print(f"Found a better match at {r} with segSim {refNodes_plSorted_segSim[r_i]}")
                        break
                if refNode == refNodes_plSorted[0]:
                    print(f"Best match {refNode} is still not reliable, stuck!")
        return refNodes_plSorted, matchPairs

    def procCurr_multiNodeControl(self,currImgIndexInRef=5,nodeIdx=0,depth=None):
        self.currDepth = self.preProcDepth(depth) if depth is not None else None
        self.update_localizer_iter_lb()
        self.process_currImg(currImgIndexInRef)
        currImg_mask_coords, currImg_mask_area, currImg_rft_dino = self.currImgInfo['currImg_mask_coords'], self.currImgInfo['currImg_mask_area'], self.currImgInfo['rft_dino']
        currImg, masks, nodesCurr = self.currImgInfo['currImg'], self.currImgInfo['masks'], self.currImgInfo['nodesCurr']
        curr_node_desc = currImg_rft_dino.copy()
        print(f"curr_node_desc.shape: {curr_node_desc.shape}")

        if self.followFloor: 
            txtFt_floor = self.get_txt_ft("floor")
            matchMat_qrySeg2txt = (self.currImgInfo['rft_clip'] @ txtFt_floor.T)[:,0]
            matchMat_qrySeg2txt = np.exp(100*matchMat_qrySeg2txt) / np.exp(100*matchMat_qrySeg2txt).sum()
        
        matchPairs = self.relocalize_replan(currImgIndexInRef,curr_node_desc,nodesCurr)[1]
        ## fast versions possible too
        # pl = self.allPathLengths[self.daNbrs[matchPairs[:,1]],self.goalNodeNbrs].min(-1)
        # pl = self.allPathLengths[matchPairs[:,1],self.goalNodeNbrs].min(-1)
        pl = []
        nodesClose2Goal = []
        for s in matchPairs[:,1]:
            # p = np.min([len(self.get_path(s,g,weight=self.mapNodeWeightStr)) for g in self.goalNodeNbrs])
            # p = np.min([self.allPathLengths[s,g] for g in self.goalNodeNbrs])
            plMinPerMatchNbr = []
            s_nbrs = [s]
            # minimize path length over DA nbrs of the matched ref node
            if self.args.plan_da_nbrs:
                s_nbrs += self.daNbrs[s]

            for s2 in s_nbrs:
                plMinPerMatchNbr.append(np.min([self.allPathLengths[s2,g] for g in self.goalNodeNbrs]))
            if len(plMinPerMatchNbr) == 0: # POSSIBLE BUG
                p = -1
                nbrClosest2goal = s
            else:
                p = np.min(plMinPerMatchNbr)
                nbrClosest2goal = s_nbrs[np.argmin(plMinPerMatchNbr)]
            nodesClose2Goal.append(nbrClosest2goal)
            pl.append(p)
        pl = np.array(pl)
        pl[pl==-1] = pl.max() + 1
        self.pathLengths.append(pl.mean())
        print(f"Path length mean: {pl.mean():.2f}, median: {np.median(pl):.2f}, min: {pl.min()}, max: {pl.max()}")
        print(f"Num matches: {matchPairs.shape[0]}")
        weights = np.ones_like(pl)
        if np.unique(pl).shape[0] == 1:
            print(f"same path length for all matches: {pl[0]}")
        else:
            weights = 1 - (pl - pl.min())/(pl.max() - pl.min())
            weights = np.exp(2*weights)
            weights = weights / weights.sum()
            # weights[pl!=pl.min()] = 0

        if self.args.fixed_lin_vel is not None:
            v = self.args.fixed_lin_vel
            areaRatio_map2qry = -1
            fwdVals = None
            areas_qry, areas_ref = None, None
        else:
            if self.currDepth is not None:
                dep_ref = self.nodes2key(nodesClose2Goal,'depth')
                dep_qry = self.nodes2key(nodesCurr[matchPairs[:,0]],'depth')
                fwdVals = dep_ref - dep_qry
                dep_diff = (weights*fwdVals).mean()
                v = dep_diff * 50 * self.gain_lin
                areaRatio_map2qry = dep_diff
            else:
                # compute fwd/bwd control
                # nodesRefG3 = [self.G.nodes[self.G3.get_contracted_node(n)] for n in matchPairs[:,1]]
                # areas_ref = self.nodes2key(nodesRefG3,'area')
                areas_ref = self.nodes2key(nodesClose2Goal,'area')
                areas_qry = self.nodes2key(nodesCurr[matchPairs[:,0]],'area')
                areasRatio = np.minimum(areas_ref,areas_qry) / np.maximum(areas_ref,areas_qry)
                areasRatioFilterdInds = areasRatio<1 #areasRatio>0.7
                areaRatio_map2qry = areas_ref/areas_qry
                fwdVals = areaRatio_map2qry
                if areasRatioFilterdInds.sum() > 0:
                    areaRatio_map2qry = (weights[areasRatioFilterdInds]*areaRatio_map2qry[areasRatioFilterdInds]).mean()
                    fwd = min(1.5,areaRatio_map2qry) * self.gain_lin/1.5
                    bwd = -min(1.5,1.0/areaRatio_map2qry) * self.gain_lin/1.5
                    v = fwd if areaRatio_map2qry > 1 else bwd
                else:
                    areaRatio_map2qry = 1.0
                    v = 0.0

        # check if done
        if pl.min() == 1:
            print(matchPairs[pl==1])
            print(f"Path length 1 for refNode {matchPairs[pl.argmin(),1]}")
            self.done = True

            # generate a visual for the final goal img
            qryNode, lastNode = matchPairs[pl.argmin()]
            lastNodeVis = func.showSegFromH5(self.h5FullPath,self.ims,self.nodeID_to_imgRegionIdx[lastNode][0],self.nodeID_to_imgRegionIdx[lastNode][1],cfg=self.cfg,dataDir=self.imgDir,doPlot=False)[:,:,::-1]
            qryNodeVis = func.get_vis_anns([nodesCurr[qryNode]],currImg.copy(),skipH5indexing=True)
            lastNodeVis = np.concatenate([lastNodeVis,qryNodeVis],axis=1)
            cv2.imwrite(f"{self.controller.folder_path}/lastNodeVis.png", lastNodeVis*255)

        coords_q = currImg_mask_coords[matchPairs[:,0]]
        if self.followFloor:
            coords_q = np.concatenate([coords_q,currImg_mask_coords],axis=0)
            weights = np.concatenate([0.1*weights,matchMat_qrySeg2txt],axis=0)
        # coords_r = self.nodes2key(matchPairs[:,1],'coords')
        colors, norm = value2color(weights,cmName='winter')
        coords_r_ = coords_q.copy()
        coords_r_[:,0] = self.img_w//2
        weightedSum = (weights[:,None] * (coords_q-coords_r_)).sum(0)
        x_off = weightedSum[0]
        w = -x_off * self.gain_ang/(self.img_w//2)
        refImgInds = self.nodeID_to_imgRegionIdx[matchPairs[:,1]][:,0]
        bc = np.bincount(refImgInds)#,weights=weights)
        self.localizedImgIdx = bc.argmax()
        print(f"Localized to imgIdx: {self.localizedImgIdx}")

        if self.args.controlModelPath is not None:
            # coords_q = currImg_mask_coords[matchPairs[:,0]]
            areas_qry = self.nodes2key(nodesCurr[matchPairs[:,0]],'area')
            pls_i = pl.copy()/100.0
            if (pls_i.max() - pls_i.min()) == 0:
                pls_i = np.zeros_like(pls_i)
            else:
                pls_i = 1 + (pls_i - pls_i.min()) / (pls_i.max() - pls_i.min())
            inputVec = np.concatenate([coords_q/np.array([self.img_w,self.img_h]) - 0.5, pls_i[:,None], areas_qry[:,None]/(self.img_w*self.img_h),  areas_qry[:,None]/(self.img_w*self.img_h), curr_node_desc[matchPairs[:,0]]], axis=1)[:,:100]
            inputVec = np.pad(inputVec, ((0, 70 - len(inputVec)), (0, 0)), mode='constant')
            inputVec = torch.tensor(inputVec,dtype=torch.float32).unsqueeze(0).cuda()
            v, w = self.controlModel(inputVec)[0].detach().cpu().numpy()[0]
            # w = -w

        # create a visual
        currImg2vis = func.get_vis_anns(nodesCurr[matchPairs[:,0]],currImg.copy(),skipH5indexing=True,drawBG=False)
        controlImg = func.drawMasksWithColors(currImg,masks[matchPairs[:,0]],colors)
        controlImg = visualize_flow(coords_q,coords_r_,controlImg,colors,norm,weights,fwdVals=fwdVals,display=False,colorbar=False).astype(float) / 255.0
        localizedRefImgNodeInds = [rn for rn in matchPairs[:,1] if self.nodeID_to_imgRegionIdx[rn][0] == self.localizedImgIdx]
        localizedRefImg = func.showSegFromH5(self.h5FullPath,self.ims,self.localizedImgIdx,self.nodeID_to_imgRegionIdx[localizedRefImgNodeInds,1],cfg=self.cfg,dataDir=self.imgDir,doPlot=False,drawBG=False)[:,:,::-1]
        txtStr = f"ImgIdx: {self.localizedImgIdx}, AreaRatio: {areaRatio_map2qry:.2f}, x_off: {x_off:.2f}, w: {w:.2f}, v: {v:.2f}"
        combinedImg = np.concatenate([localizedRefImg,currImg2vis,controlImg],axis=1)
        cv2.putText(combinedImg, txtStr, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(f"{self.controller.matchings_path}/combinedImg_{self.liveImgIter:05}.png", combinedImg*255)

        self.learnersDict.append( {'coords_q':coords_q, 'pl':pl, 'areas_q':areas_qry, 'areas_r':areas_ref}
        )

        return v,w,x_off,1,combinedImg

    def get_control_signal(self,img,depth=None):
        print(f"\n==========================\nImg: {self.liveImgIter}")
        if self.control_mode == "CC":
            retVals = self.procCurr_multiNodeControl(np.array(img),self.liveImgIter,depth=depth)
            self.liveImgIter += 1
            return retVals
        else:
            response_dict, info = self.controller.predict(np.array(img),self.liveImgIter,None)
            mapNodeDict, qryNodeDict, combinedImg, planNodes = info
            self.done = bool(response_dict['complete'])
            self.hop = bool(response_dict['hop'])

            self.liveImgIter += 1
            localized = bool(response_dict['segment_found'])
            areaRatio_map2qry = response_dict['areaRatio_map2qry']
            if localized:
                self.trackFail = False
                x_off = response_dict['u']-(self.img_w//2)
                w = -x_off * self.gain_ang/(self.img_w//2)
                fwd = max(1.5,areaRatio_map2qry) * self.gain_lin/1.5
                bwd = -max(1.5,1.0/areaRatio_map2qry) * self.gain_lin/1.5
                v = fwd if areaRatio_map2qry > 1 else bwd
                # v = self.gain_lin if areaRatio_map2qry > 1 else -self.gain_lin
                # current best matching ref image index
                self.localizedImgIdx = mapNodeDict['imgIdx'][0]
            else:
                assert(0) # relocalization performed in procCurr() now
                self.trackFail = True
                v,w,x_off = 0., 0., 0.
                refNodes_plSorted = self.relocalize_replan(img,qryNodeDict['curr_node_desc'],self.currImgInfo['nodesCurr'])[0]
                refNode = refNodes_plSorted[0]

                # segment descriptor matching
                segMatch = False
                if segMatch:
                    ref_imgInds2search = np.arange(self.localizedImgIdx-2,self.localizedImgIdx+3)
                    ref_nodes2search = [i for i in range(len(self.nodeID_to_imgRegionIdx)) if self.nodeID_to_imgRegionIdx[i][0] in ref_imgInds2search]
                    qry_descs = qryNodeDict['curr_node_desc']
                    ref_descs = self.rft_da_env_arr[ref_nodes2search]
                    matchMat_qq = qry_descs @ qry_descs.T
                    matchMat_qr = qry_descs @ ref_descs.T
                    singleBestMatch, matches_ij, margin_ij, margin_ij_sortedInds = find_matches(matchMat_qr,np.sort(matchMat_qq,1))
                    pls = np.array(pls)[matches_ij]
                    pls[pls>pls.mean()] = pls.max()
                    pls = (1-(pls - pls.min())/(pls.max()-pls.min()))
                    weights = margin_ij * pls

                    plt.plot(pls,label="Path Length")
                    plt.plot(margin_ij,label="Margin")
                    plt.plot(weights,label="Weight")
                    plt.legend()
                    plt.show()

                    colors, norm = value2color(weights)
                    currImg = self.currImgInfo['currImg'][:,:,::-1]
                    qryCords = self.currImgInfo["currImg_mask_coords"]
                    refCords = qryCords.copy()
                    refCords[:,0] = 160
                    diff = qryCords - refCords
                    weightedSum = (weights[:,None] * diff).sum(0)
                    w = -weightedSum[0] * self.gain_ang/(self.img_w//2)

                    visualize_flow(qryCords,refCords,currImg,colors,norm,weights)

                    refNode = np.array(ref_nodes2search)[matches_ij][singleBestMatch]

                self.localizedImgIdx = self.nodeID_to_imgRegionIdx[refNode][0]
                newPlan = self.get_path(refNode,self.goalNode,self.G4,'margin')
                self.controller.iter += 1
                self.plan = self.plan[:self.controller.iter] + newPlan
            return v,w,x_off,response_dict['hop']

    def init_controller(self,currImgIndexInRef=5):
        # self, currImgIndexInRef = RH, 5
        targetNode = 651 # RH.search_with_text("painting of a woman")
        refImgLoader = func.MetaIterator(lambda idx: func.loadPreProcImgFromIdx(idx,self.imgDir,self.ims,self.cfg))

        currImgInfo = self.process_currImg(currImgIndexInRef)
        currImg_mask_coords, currImg_mask_area, currImg_rft_dino = currImgInfo['currImg_mask_coords'], currImgInfo['currImg_mask_area'], currImgInfo['rft_dino']
        currImg, masks, nodesCurr = currImgInfo['currImg'], currImgInfo['masks'], currImgInfo['nodesCurr']

        subMapInfo = self.getSubMap(currImgIndexInRef)
        nodeIndsFromRef = subMapInfo['nodeIndsFromRef']
        rft_dino_ref = subMapInfo['rft_dino_ref']

        matchMat = currImg_rft_dino @ rft_dino_ref.T
        matchScores = np.max(matchMat,1)
        matchInds = np.argmax(matchMat,1)
        # mutual NN check
        matchScores_ref2Qry = np.max(matchMat,0)
        matchInds_ref2Qry = np.argmax(matchMat,0)
        mutualNN = np.argwhere(matchInds_ref2Qry[matchInds] == np.arange(len(matchInds))).flatten()
        matchScores = matchScores[mutualNN]
        matchInds = matchInds[mutualNN]
        maskNodes = [nodesCurr[i] for i in mutualNN]
        currImg_mask_coords = currImg_mask_coords[mutualNN]
        currImg_mask_area = currImg_mask_area[mutualNN]
        masks = masks[mutualNN]
        print(f"matchInds.shape: {matchInds.shape}")

        # measure aliasing
        currImg_selfSim = currImg_rft_dino @ currImg_rft_dino.T
        nn1_matchScore= np.sort(currImg_selfSim[mutualNN],-1)[:,-2]

        # measure path lengths
        matchedNodes_contracted = [self.G3.get_contracted_node(n) for n in matchInds]
        pls = []
        for s in matchedNodes_contracted:
            pl = nx.shortest_path_length(self.G3.graph,s,target=targetNode)
            pls.append(pl)
        pls = np.array(pls)
        
        # compute qry node weights
        weights = np.maximum(0,matchScores-nn1_matchScore) * (1-(pls - pls.min())/(pls.max()-pls.min()))

        colors, norm = value2color(weights)
        refNodes_mask_coords = getCoordsFromNodes(self.G,nodeIndsFromRef)
        visualize_flow(currImg_mask_coords,refNodes_mask_coords[matchInds],currImg,colors,norm)

        areaSortingInds = np.argsort(currImg_mask_area)[::-1]
        pathHeatMap = func.drawMasksWithColors(currImg,masks[areaSortingInds],colors[areaSortingInds])
        plt.imshow(pathHeatMap)
        plt.show()

        plt.imshow(func.drawMasksWithColors(currImg,masks[areaSortingInds],value2color(matchScores)[0][areaSortingInds]))
        plt.show()
        
        # nodeIdx = nodeIndsFromRef[matchScores_ref2Qry.argsort()[::-1][5:10]]
        # for nix in nodeIdx:
        #     print(nix)
        #     _ = func.showSegFromH5(RH.h5FullPath,RH.ims,RH.nodeID_to_imgRegionIdx[nix][0],RH.nodeID_to_imgRegionIdx[nix][1],RH.cfg,dataDir=RH.imgDir)
        # RH.search_with_text("painting of a woman")
        # path = RH.get_path(204,651)
        # RH.display_path_images(path)

        return

    def get_txt_ft(self,txtQry):
        mask_generator, dino, clip, tokenizer, clip_preprocess = self.models
        txt_ft = func.getCLIPtxtFeat(txtQry,tokenizer,clip).detach().cpu().numpy()
        return txt_ft
        
    def query_with_text(self,masks,rft_clip,txtQry="floor",txt2img_thresh=0.22,displayImg=None):
        txt_ft = self.get_txt_ft(txtQry)
        print(f"txt_ft.shape (numQueries x ftDim): {txt_ft.shape}")

        matchMat = rft_clip @ txt_ft.T
        matchedNodes = np.argsort(-matchMat,0).flatten()
        sim_imgTxt = matchMat[matchedNodes].flatten()
        matchedNodes_thresh = [matchedNodes[i] for i in range(len(matchedNodes)) if sim_imgTxt[i] > txt2img_thresh]
    
        masks_txt = [masks[n] for n in matchedNodes_thresh]
        if displayImg is not None:
            vis = func.get_vis_anns(masks_txt,displayImg.copy(),skipH5indexing=True,drawBG=False)
            plt.imshow(vis)
            plt.show()
        return masks_txt, matchedNodes_thresh

    def bp(self,currImgIndexInRef=5):
        # self, currImgIndexInRef = RH, 5
        from libs.belief_propagation import factor_graph as FG

        currImgInfo = self.process_currImg(currImgIndexInRef)
        currImg_mask_coords, currImg_mask_area, currImg_rft_dino = currImgInfo['currImg_mask_coords'], currImgInfo['currImg_mask_area'], currImgInfo['rft_dino']
        currImg, masks = currImgInfo['currImg'], currImgInfo['masks']

        subMapInfo = self.getSubMap(currImgIndexInRef)
        nodeIndsFromRef = subMapInfo['nodeIndsFromRef']
        rft_dino_ref = subMapInfo['rft_dino_ref']

        matchMat = currImg_rft_dino @ rft_dino_ref.T

        edgeList, G_curr = func.getSingleImageGraph(currImg_mask_coords,currImg_mask_area)
        repInds_ij = check_repeated_coords(currImg_mask_coords)
        for ij in repInds_ij:
            for e in edgeList:
                if ij[1] in e:
                    e_ = e.copy()
                    e_[e_==ij[1]] = ij[0]
                    G_curr.add_edges_from([e_])
        edgeList = np.array(G_curr.edges)

        refPathLengths = np.empty([len(nodeIndsFromRef),len(nodeIndsFromRef)],float)
        for i,n1 in enumerate(nodeIndsFromRef):
            for j,n2 in enumerate(nodeIndsFromRef):
                if i == j:
                    l = 0
                else:
                    l = len(nx.shortest_path(self.G,n1,n2))
                refPathLengths[i,j] = l
        refPathLengths = refPathLengths.max() - refPathLengths

        binaryFactors = []
        for i,e in enumerate(edgeList):
            fac = FG.factor([f'{e[0]}', f'{e[1]}'], refPathLengths)
            binaryFactors.append(fac)
        unaryFactors = []
        for i,n in enumerate(G_curr.nodes):
            fac = FG.factor([f'{n}'],matchMat[i])
            unaryFactors.append(fac)

        mrf = FG.factor_graph()
        factorsAll = binaryFactors + unaryFactors
        for i,f in enumerate(factorsAll):
            mrf.add_factor_node(f"f_{i}",f)

        # bp = FG.belief_propagation(mrf)
        lbp = FG.loopy_belief_propagation(mrf)
        matchInds = [lbp.belief(f'{i}', 30).get_distribution().argmax() for i in range(matchMat.shape[0])]

        nodeIndsFromRef_matched = nodeIndsFromRef[matchInds]
        imgInds = np.unique(self.nodeID_to_imgRegionIdx[nodeIndsFromRef_matched][:,0])
        for ii in imgInds:
            matchIndsSub = np.argwhere(self.nodeID_to_imgRegionIdx[nodeIndsFromRef_matched][:,0] == ii).flatten()
            
            print(f"num matches from imgIdx {ii}: {len(matchIndsSub)}")
            _ = func.showSegFromH5(self.h5FullPath,self.ims,ii,self.nodeID_to_imgRegionIdx[nodeIndsFromRef_matched][matchIndsSub][:,1],self.cfg,dataDir=self.imgDir,drawBG=False)
            
            plt.imshow(func.drawMasksWithColors(currImg,masks[matchIndsSub],value2color(np.arange(len(matchIndsSub)))[0]))
            plt.show()
            ## plot individual matches
            # for mm in matchIndsSub:
            #     _ = func.showSegFromH5(self.h5FullPath,self.ims,ii,self.nodeID_to_imgRegionIdx[nodeIndsFromRef_matched][mm][1],self.cfg,dataDir=self.imgDir,drawBG=False)
                
            #     plt.imshow(func.drawMasksWithColors(currImg,[masks[mm]],value2color(np.arange(1))))
            #     plt.show()

    def showNode(self,nodeIdx,**kwargs):
        imgIdx, regIdx = self.nodeID_to_imgRegionIdx[nodeIdx]
        _ = func.showSegFromH5(self.h5FullPath,self.ims,imgIdx,regIdx,self.cfg,dataDir=self.imgDir)

class Agent_GNM():
    def __init__(self,modelconfigpath,modelweightspath,modelname,mapdir,precomputedDir=None,goal_node=-1,radius=2,device=torch.device('cuda')):
        self.device = device
        self.modelname = modelname
        # self.context_queue = context_queue
        self.distances = []
        self.waypoints = []

        self.model, self.model_params = self.ready_model(modelconfigpath,modelweightspath,self.modelname)
        self.topomap = self.load_map(mapdir)
        self.localmapIdx = 2
        self.localmap1D = self.topomap[self.localmapIdx-2:self.localmapIdx+3]
        self.currImgHistory = []

        if precomputedDir is None:
            self.precomputedDir = "./out/cache/GNM/"
        else:
            self.precomputedDir = precomputedDir
        os.makedirs(self.precomputedDir, exist_ok=True)
        self.map_distance_matrix = self.create_adjMat()
        self.map_graph = self.create_graph_from_adjMat()

        # TODO: below is underworks
        closest_node = 0
        assert -1 <= goal_node < len(self.topomap), "Invalid goal index"
        if goal_node == -1:
            goal_node = len(self.topomap) - 1
        else:
            goal_node = goal_node
        self.reached_goal = False
        self.start = max(closest_node - radius, 0)
        self.end = min(closest_node + radius + 1, goal_node)

    def maintain_history(self,obs_img):
        diff = len(self.currImgHistory) - self.model_params[self.modelname]["context"] - 1
        if diff < 0:
            for _ in range(abs(diff)):
                self.currImgHistory.append(obs_img)
        else:
            self.currImgHistory.pop(0)
            self.currImgHistory.append(obs_img)

    def updateLocalMap(self,localizationIdx_local):
        ptr = localizationIdx_local + self.localmapIdx
        self.localmap1D = self.topomap[ptr-2:ptr+3]
        self.localmapIdx = ptr

    def ready_model(self,modelconfigpath,modelweightspath,modelname):
        model_params = {}
        # load model parameters
        with open(modelconfigpath, "r") as f:
            model_config = yaml.safe_load(f)
        for param in model_config:
            model_params[param] = model_config[param]

        # load model weights
        model_filename = model_config[modelname]["path"]
        model_path = os.path.join(modelweightspath, model_filename)
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        model = load_model(
            model_path,
            model_params[modelname]["model_type"],
            model_params[modelname]["context"],
            model_params[modelname]["len_traj_pred"],
            model_params[modelname]["learn_angle"], 
            model_params[modelname]["obs_encoding_size"], 
            model_params[modelname]["goal_encoding_size"],
            device=self.device,
        )
        model.eval()
        return model, model_params

    def load_map(self,mapdir):
        imgnames = sorted(os.listdir(mapdir), key=lambda x: int(x.split(".")[0]))
        num_nodes = len(os.listdir(mapdir))
        topomap = []
        for i in range(num_nodes):
            image_path = os.path.join(mapdir, imgnames[i])
            topomap.append(PILImage.open(image_path))
        return topomap

    def create_adjMat(self):
        if os.path.exists(os.path.join(self.precomputedDir, "distance_matrix.npy")):
            print("Loading precomputed adjacency matrix...")
            return np.load(os.path.join(self.precomputedDir, "distance_matrix.npy"))
        else:
            print("No precomputed adjacency matrix found, computing...")
        # take all map images and predict distances all-vs-all
        l = self.model_params[self.modelname]["context"]
        dists = []
        for i, currImg in enumerate(tqdm(self.topomap)):
            if i < l:
                dists.append([])
                continue
            else:
                currImgHistory = self.topomap[i-l:i+1]
                currImgHistory = transform_images(currImgHistory, self.model_params[self.modelname]["image_size"])
            distsPerImg = []
            for goalImg in self.topomap:
                goalImg = transform_images(goalImg, self.model_params[self.modelname]["image_size"])
                dist, waypoint = self.model(currImgHistory.to(self.device), goalImg.to(self.device)) 
                distsPerImg.append(to_numpy(dist[0]))
                self.waypoints.append(to_numpy(waypoint[0]))
            dists.append(np.array(distsPerImg))
        # copy initial lth row to 0 to l-1
        for i in range(l):
            dists[i] = dists[l]
        dists = np.array(dists).squeeze()
        # save distance matrix
        np.save(os.path.join(self.precomputedDir, "distance_matrix.npy"), dists)
        return dists

    def create_graph_from_adjMat(self,minDis=15):
        # create nx graph from distance matrix
        # TODO: create a DiGraph instead
        G = nx.from_numpy_array(self.map_distance_matrix)
        # remove edges with distance > minDis
        for i in range(len(self.map_distance_matrix)):
            for j in range(len(self.map_distance_matrix)):
                if self.map_distance_matrix[i,j] < minDis:
                    if [i,j] in G.edges():
                        G.remove_edge(i,j)
        # shortest path can be found as
        # nx.shortest_path(G,10,100,weight='weight')
        return G

    # TODO: underworks
    def obs_to_act(self,img):
        transf_obs_img = transform_images(img, self.model_params["image_size"])
        for map_img in self.topomap[self.start: self.end + 1]:
            transf_map_img = transform_images(map_img, self.model_params["image_size"])
            dist, waypoint = self.model(transf_obs_img, transf_map_img) 
            self.distances.append(to_numpy(dist[0]))
            self.waypoints.append(to_numpy(waypoint[0]))

    def predict_currHistAndGoal(self,currImgHistory,goalImg):
        currImgHistory = transform_images(currImgHistory, self.model_params[self.modelname]["image_size"])
        goalImg = transform_images(goalImg, self.model_params[self.modelname]["image_size"])
        dist, waypoint = self.model(currImgHistory.to(self.device), goalImg.to(self.device)) 
        return to_numpy(dist), to_numpy(waypoint)

    def waypoint_to_velocity(self,waypoint,agent_params,time_step):
        EPS = 1e-8
        max_v, max_w = agent_params["max_v"], agent_params["max_w"]

        """PD controller for the robot"""
        assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
        if len(waypoint) == 2:
            dx, dy = waypoint
        else:
            dx, dy, hx, hy = waypoint
        # this controller only uses the predicted heading if dx and dy near zero
        if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
            v = 0
            w = clip_angle(np.arctan2(hy, hx))/time_step
        elif np.abs(dx) < EPS:
            v =  0
            w = np.sign(dy) * np.pi/(2*time_step)
        else:
            v = dx / time_step
            w = np.arctan(dy/dx) / time_step
        v = np.clip(v, 0, max_v)
        w = np.clip(w, -max_w, max_w)
        return v, w

def clip_angle(theta) -> float:
    """Clip angle to [-pi, pi]"""
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi

def value2color(values,vmin=None,vmax=None,cmName='jet'):
    cmapPaths = matplotlib.colormaps.get_cmap(cmName)
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = np.array([cmapPaths(norm(value))[:3] for value in values])
    return colors, norm

def visualize_flow(cords_org,cords_dst,img=None,colors=None,norm=None,weights=None,cmap='jet',colorbar=True,display=True,fwdVals=None):

    diff = cords_org - cords_dst
    dpi = 100
    img_height, img_width = img.shape[:2]  # Get the image dimensions
    fig_width, fig_height = img_width / dpi, img_height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    if img is not None: ax.imshow(img)
    # for i in range(len(currImg_mask_coords)):
    #     ax.plot(currImg_mask_coords[i,0],currImg_mask_coords[i,1],'o',color='r')
    #     ax.plot(refNodes_mask_coords[matchInds[i],0],refNodes_mask_coords[matchInds[i],1],'o',color='b')
    if fwdVals is not None:
        # plot a diamond for negative values and a circle for positive values, size = val
        pointTypeMask = fwdVals > 0
        ax.scatter(*(cords_org[pointTypeMask].T),c=colors[pointTypeMask],s=abs(fwdVals[pointTypeMask])*40,marker='o',edgecolor='white',linewidth=0.5)
        ax.scatter(*(cords_org[~pointTypeMask].T),c=colors[~pointTypeMask],s=abs(abs(fwdVals[~pointTypeMask]))*40,marker='X',edgecolor='white',linewidth=0.5)
    if weights is not None:
        weightedSum = (weights[:,None] * diff).sum(0)
        ax.quiver(*(np.array([160,120]).T), weightedSum[0], weightedSum[1],color='black',edgecolor='white',linewidth=0.5)
    ax.quiver(*(cords_org.T), diff[:,0], diff[:,1],color=colors,edgecolor='white',linewidth=0.5)
    if colorbar: add_colobar(ax,plt,norm,cmap)
    ax.set_xlim([0, img_width])
    ax.set_ylim([img_height,0])
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if display:
        plt.show()
    else:
        # return the figure as image (same size as img imshow-ed above)
        fig.canvas.draw()
        vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        vis = cv2.resize(vis,(img.shape[1],img.shape[0]))
        plt.close(fig)
        return vis

def add_colobar(ax,plt,norm=None,cmap='jet'):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create a ScalarMappable object with the "autumn" colormap
    if norm is None:
        norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Add a colorbar to the axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")

    # Customize the colorbar
    cbar.set_label('Colorbar Label', labelpad=10)
    cbar.ax.yaxis.set_ticks_position('right')

def getCoordsFromNodes(G=None,nodes=None):
    assert(G is not None or nodes is not None)
    if G is not None:
        if nodes is None:
            nodes = G.nodes()
        else:
            nodes = [G.nodes[n] for n in nodes]
    coords = np.array([np.array(np.nonzero(n['segmentation'])).mean(1)[::-1].astype(int) for n in nodes])
    return coords

def check_repeated_coords(coords):
    cDiff = abs(coords[:,None,:] - coords[None,:,:]).sum(-1)
    inds_ij = np.argwhere(np.triu(cDiff+1,1) == 1)
    return inds_ij

def find_matches(mat_ij,sortedSim_ii):
    sortedSim_ij = np.sort(mat_ij,1)
    margin_ij = np.maximum(0, sortedSim_ij[:,-1] - np.maximum(sortedSim_ij[:,-2],sortedSim_ii[:,-2]))
    matches_ij = np.argmax(mat_ij,1)
    margin_ij_sortedInds = np.argsort(-margin_ij)
    singleBestMatch = margin_ij_sortedInds[0]
    return singleBestMatch, matches_ij, margin_ij, margin_ij_sortedInds

def create_edges_DT(mask_cords):
    if len(mask_cords) > 3:
        tri = Delaunay(mask_cords)
        nbrs, nbrsLists = [], []
        for v in range(len(mask_cords)):
            nbrsList = func.getNbrsDelaunay(tri, v)
            nbrsLists.append(nbrsList)
            nbrs += nbrsList
        nbrs = func.removeDuplicateNbrPairs(nbrs)
    else:
        numCords = len(mask_cords)
        nbrs = [[u,v] for u in range(numCords) for v in range(u + 1, numCords)]
    return np.array(nbrs)

def plot_DT():
    nodeIndsPerImg = getNodeInds(RH,0)
    nodeDepth = 100*self.nodes2key(nodeIndsPerImg,'depth')
    nodeCoords = self.nodes2key(nodeIndsPerImg,'coords')
    nodeCoords3d = np.concatenate((nodeCoords,nodeDepth[:,None]),axis=1)
    nbrs = AA.create_edges_DT(nodeCoords3d[::-1])

    plt.imshow(RH.getImg(0))
    plt.plot(*np.vstack([coords[nbrs[:,0]], coords[nbrs[:,1]]]).T, 'r')
    plt.triplot(coords[:,0], coords[:,1], AA.func.Delaunay(coords).simplices)

def plot3d(c3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(c3d[:,0], c3d[:,1], c3d[:,2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

