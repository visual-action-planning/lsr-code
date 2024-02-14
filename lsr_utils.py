# Implementation of Latent space roadmap functions for the paper :
# Latent Space Roadmap for Visual Action Planning of Deformable and Rigid Object Manipulation
#-----------------------------------------------------------------------------------------------

import argparse
import os
import sys
from importlib.machinery import SourceFileLoader
import algorithms as alg
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from dataloader import TripletTensorDataset
import architectures.VAE_ResNet as vae
import cv2
import pickle
import networkx as nx
import random

#Find closesd nodes in graph G using distance type (1, 2, np.inf)
def get_closest_nodes(G, z_pos_c1, z_pos_c2, distance_type=2):
    c1_close_idx=-1
    c2_close_idx=-1
    min_distance_c1=np.Inf
    min_distance_c2=np.Inf

    #find the closest nodes
    for g in G.nodes:
        tz_pos=G.nodes[g]['pos']
        node_distance_c1=np.linalg.norm(z_pos_c1-tz_pos, ord=distance_type)
        node_distance_c2=np.linalg.norm(z_pos_c2-tz_pos, ord=distance_type)
        if node_distance_c1<min_distance_c1:
            min_distance_c1=node_distance_c1
            c1_close_idx=g

        if node_distance_c2<min_distance_c2:
            min_distance_c2=node_distance_c2
            c2_close_idx=g

    return c1_close_idx, c2_close_idx


#Format distance type from string to correct format
def format_distance_type(distance_type):
    if distance_type=='inf' or distance_type==np.inf:
        return np.inf
    else:
        return int(distance_type)



#Implemention of the Latent Space Roadmap
def build_lsr(latent_map_file,epsilon,distance_type,graph_name, config_file,checkpoint_file,min_edge_w=0,min_node_m=0, directed_graph = False,  hasclasses=False, save_node_imgs=False, verbose = False, save_graph = False):

    #load VAE
    vae_config_file = os.path.join('.', 'configs', config_file + '.py')
    vae_directory = os.path.join('.', 'models', checkpoint_file)

    vae_config = SourceFileLoader(config_file, vae_config_file).load_module().config
    vae_config['exp_name'] = config_file
    vae_config['vae_opt']['exp_dir'] = vae_directory # the place where logs, models, and other stuff will be stored
    print(' *- Loading config %s from file: %s' % (config_file, vae_config_file))

    vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
    print(' *- Loaded {0}'.format(vae_config['algorithm_type']))

    vae_algorithm.load_checkpoint('models/'+config_file+"/"+checkpoint_file)
    vae_algorithm.model.eval()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    distance_type=format_distance_type(distance_type)


    graph_base_path = "graphs"

    if not os.path.exists(graph_base_path):
        os.mkdir(graph_base_path)
    graph_path = graph_base_path+"/"+graph_name

    # load latent data
    f = open(latent_map_file, 'rb')
    latent_map = pickle.load(f)
    len_latent_map = len(latent_map)


    #Build the Graph
    #Phase 1 ***************************************
    if directed_graph:
        G1 = nx.DiGraph()
    else:
        G1=nx.Graph()
    #1.1 build all nodes
    counter=0
    Z_all=set()
    for latent_pair in latent_map:
        counter+=1
        if verbose:
            print("checking " + str(counter)+ " / " + str(len_latent_map)+ " build " + str(G.number_of_nodes()) + " so far.")
        if not hasclasses:
            # get the latent coordinates
            z_pos_c1=latent_pair[0]
            z_pos_c2=latent_pair[1]
            action=latent_pair[2]
        else:
            #c1_str, z_np, c2_str, z_np2,action
            z_pos_c1=latent_pair[1]
            z_pos_c2=latent_pair[3]
            action=latent_pair[4]
        #action pairs
        dis=np.linalg.norm(z_pos_c1-z_pos_c2,ord=distance_type)
        if action==1:
            c_idx=G1.number_of_nodes()
            G1.add_node(c_idx,pos=z_pos_c1)
            Z_all.add(c_idx)
            c_idx=G1.number_of_nodes()
            G1.add_node(c_idx,pos=z_pos_c2)
            Z_all.add(c_idx)
            G1.add_edge(c_idx-1,c_idx,l=np.round(dis,1))
        #no action
        if action==0:# and dis<epsilon:
            c_idx=G1.number_of_nodes()
            G1.add_node(c_idx,pos=z_pos_c1)
            Z_all.add(c_idx)
            c_idx=G1.number_of_nodes()
            G1.add_node(c_idx,pos=z_pos_c2)
            Z_all.add(c_idx)

    #Print result of phase 1
    print("***********Phase one done*******")
    print("Num nodes: " + str(G1.number_of_nodes()))
    print("Num edges: " + str(G1.number_of_edges()))
    print("num in Z_all: " + str(len(Z_all)) )
    #Phase 2**********************************************************
    #speedup
    H1 = G1.copy()
    Z_sys_is=[]
    #2.6 #while not Z_noachtion= null
    while len(Z_all) >0:
        print("Z_all size: " + str(len(Z_all)))
        #2.1 randomly select z E Z_noaction
        z = random.choice(tuple(Z_all))
        #2.2 first time
        W_z=set()
        #z also belongs to the set
        W_z.add(z)
        #2.3 for all w E W_wz find W_w from 2.2 and set W_z:=W_z U W_w
        s_len_wz=len(W_z)
        #set init end length
        e_len_wz=np.Inf
        #check speedup
        W_w_to_check=W_z.copy()        
        #W_w_to_check=W_z
        while not s_len_wz == e_len_wz:
            s_len_wz=len(W_z)
            #2.2 find all w E G1 for wich ||z-w||_d < 2*epsilon
            W_w=set() 
            for w in W_w_to_check:
                w_pos=H1.nodes[w]['pos']
                for wn in H1:
                    wn_pos=H1.nodes[wn]['pos']
                    dis=np.linalg.norm(wn_pos-w_pos,ord=distance_type)
                    #could be smaler to be more conservative
                    if dis < epsilon:
                        W_w.add(wn)
            #check speedup
            for w in W_w_to_check:
                H1.remove_node(w)
            W_w_to_check=W_w-W_w_to_check
            W_z=W_z.union(W_w)
            e_len_wz=len(W_z)            
            #for speedup remove W_z from G1 copy
        #2.4 Z_noaction:=Z_noaztipn - W_z
        Z_all=Z_all - W_z
        #2.5 append W_z to a list
        Z_sys_is.append(W_z)

    #Print result of phase 2
    print("***********Phase two done*******")
    print("Num disjoint sets: " + str(len(Z_sys_is)))
    num_z_sys_nodes=0
    w_z_min=np.Inf
    w_z_max=-np.Inf
    for W_z in Z_sys_is:
        if len(W_z)<w_z_min:
            w_z_min=len(W_z)
        if len(W_z) > w_z_max:
            w_z_max=len(W_z)
        num_z_sys_nodes+=len(W_z)
    print("Total number of components: " + str(num_z_sys_nodes))
    print("Max number W_z: " + str(w_z_max)+ " min number w_z: " + str(w_z_min))

    # Phase 3 ********************************************************
    if directed_graph:
        G2 = nx.DiGraph()
    else:
        G2=nx.Graph()
    #3.1 build centroids-W_z nodes
    for W_z in Z_sys_is:
        w_pos_all=[]
        for w in W_z:
            w_pos=G1.nodes[w]['pos']
            w_pos_all.append(w_pos)
        W_z_c_pos=np.mean(w_pos_all,axis=0)
        #decode image
        z_pos=torch.from_numpy(W_z_c_pos).float().to(device)
        z_pos=z_pos.unsqueeze(0)
        img_rec,_=vae_algorithm.model.decoder(z_pos)
        img_rec_cv=img_rec[0].detach().permute(1,2,0).cpu().numpy()
        c_idx=G2.number_of_nodes()
        if save_node_imgs:
            G2.add_node(c_idx,pos=W_z_c_pos,image=img_rec_cv,W_z=W_z,w_pos_all=w_pos_all)
        else:
            G2.add_node(c_idx,pos=W_z_c_pos,W_z=W_z,w_pos_all=w_pos_all)

    #3.2 build edges 
    #each node in the graph 2
    for g2 in G2:
        print(str(g2)+ " / " + str(G2.number_of_nodes()))
        #W_z hold components of nodes
        W_z=G2.nodes[g2]['W_z']
        #for each component
        for w in W_z:
            #find the partner
            w_pairs=G1.neighbors(w)
            for w_pair in w_pairs:
                for g2_again in G2:
                    W_z_again=G2.nodes[g2_again]['W_z']
                    for w_again in W_z_again:
                        #find the node
                        if w_again ==w_pair:
                            dis=np.linalg.norm(G2.nodes[g2_again]['pos']-G2.nodes[g2]['pos'],ord=distance_type)
                            if not G2.has_edge(g2,g2_again):
                                G2.add_edge(g2,g2_again,l=np.round(dis,1),ew=1)
                                if verbose:
                                    print("Num edges: "+str(G2.number_of_edges()))
                            else:
                                #update edge
                                ew=G2.edges[g2, g2_again]['ew']
                                l=G2.edges[g2, g2_again]['l']
                                G2.edges[g2, g2_again]['l']=(ew*l+dis)/(ew+1)
                                ew+=1
                                G2.edges[g2, g2_again]['ew']=ew

    print("***********Phase three done*******")
    print("Num nodes: " + str(G2.number_of_nodes()))
    print("Num edges: " + str(G2.number_of_edges()))


    #phase 4 Pruning
    print("Pruning edges with ew < " + str(min_edge_w))
    num_edges=G2.number_of_edges()
    remove_edges=[]
    for edge in G2.edges:
        sidx=edge[0]
        gidx=edge[1]
        ew=G2.edges[sidx, gidx]['ew']
        if ew < min_edge_w:
            remove_edges.append((sidx,gidx))
    for re in remove_edges:
        G2.remove_edge(re[0],re[1])
    num_edges_p=G2.number_of_edges()
    if num_edges > 0:
        print("pruned " + str(num_edges-num_edges_p) + " edges ( " + str(100-(num_edges_p*100.)/num_edges) + " %")
    else:
        print("pruning: num_edges = 0")

    #pruine weak nodes
    print("Pruning nodes with mearges < " + str(min_node_m))
    num_nodes=G2.number_of_nodes()
    remove_nodes=[]
    for g in G2.nodes:
        ngm=G2.nodes[g]['w_pos_all']
        if len(ngm) < min_node_m:
            remove_nodes.append(g)

    for re in remove_nodes:
        G2.remove_node(re)

    num_nodes_p=G2.number_of_nodes()
    if num_nodes > 0:
        print("pruned " + str(num_nodes-num_nodes_p) + " nodes ( " + str(100-(num_nodes_p*100.)/num_nodes) + " %")

    #prune single nodes
    num_nodes=G2.number_of_nodes()
    remove_nodes=[]
    isolates=nx.isolates(G2)
    for iso in isolates:
        remove_nodes.append(iso)

    for re in remove_nodes:
        G2.remove_node(re)

    print("pruned " + str(num_nodes-G2.number_of_nodes()) +" isolated nodes")


    print("final Graph ************************************")
    print("Num nodes: " + str(G2.number_of_nodes()))
    print("Num edges: " + str(G2.number_of_edges()))

    if save_graph:
        with open(graph_path+".pkl", 'wb') as f:
            pickle.dump(G2, f, pickle.HIGHEST_PROTOCOL)
        #nx.write_gpickle(G2, graph_path+".pkl")
        print("SAVED")
    stats_dict = {'num_nodes':num_nodes_p}
    return G2, stats_dict



#compute mean and std of action or no action pairs
def compute_mean_and_std_dev(latent_map_file,distance_type, hasclasses=False,action_mode=0):

    f = open(latent_map_file, 'rb')
    latent_map = pickle.load(f)
    len_latent_map = len(latent_map)

    distance_type=format_distance_type(distance_type)
    dist_list = []
    for latent_pair in latent_map:
        if not hasclasses:
            # get the latent coordinates
            z_pos_c1=latent_pair[0]
            z_pos_c2=latent_pair[1]
            action=latent_pair[2]
        else:
            #c1_str, z_np, c2_str, z_np2,action
            z_pos_c1=latent_pair[1]
            z_pos_c2=latent_pair[3]
            action=latent_pair[4]

        if action_mode==0:
            if action == 0:
                current_distance=np.linalg.norm(z_pos_c1-z_pos_c2, ord=distance_type)
                dist_list.append(current_distance)

        if action_mode==1:
            if action == 1:
                current_distance=np.linalg.norm(z_pos_c1-z_pos_c2, ord=distance_type)
                dist_list.append(current_distance)


    mean_dist_no_ac = np.mean(dist_list)
    std_dist_no_ac = np.std(dist_list)
    return mean_dist_no_ac, std_dist_no_ac, dist_list



#Label latent space given VAE and dataset
def lable_latent_space(config_file,checkpoint_file,output_file,dataset_name):

    #load VAE
    vae_config_file = os.path.join('.', 'configs', config_file + '.py')
    vae_directory = os.path.join('.', 'models', checkpoint_file)
    vae_config = SourceFileLoader(config_file, vae_config_file).load_module().config
    vae_config['exp_name'] = config_file
    vae_config['vae_opt']['exp_dir'] = vae_directory # the place where logs, models, and other stuff will be stored
    #print(' *- Loading config %s from file: %s' % (config_file, vae_config_file))
    vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
    #print(' *- Loaded {0}'.format(vae_config['algorithm_type']))
    num_workers = 1#vae_config_file['vae_opt']['num_workers']
    data_test_opt = vae_config['data_train_opt']

    f = open('datasets/'+dataset_name+'.pkl', 'rb')
    dataset = pickle.load(f) 


    vae_algorithm.load_checkpoint('models/'+config_file+"/"+checkpoint_file)
    vae_algorithm.model.eval()
   

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_map=[]   
    for i in range(len(dataset)):
        t=dataset[i]
        img1 = torch.tensor(t[0]/255.).float().permute(2, 0, 1)
        img2 = torch.tensor(t[1]/255).float().permute(2, 0, 1)
        img1=img1.unsqueeze_(0)
        img2=img2.unsqueeze_(0)

        ac=torch.tensor(t[2]).float()
        ac=ac.unsqueeze_(0)
        #data in pkl file are images filename. In order for the VAE to work, the images must be converted in RGB and converted into range [0,1]
        img1 = img1.to(device)
        img2 = img2.to(device)
        ac = ac.to(device)

        dec_mean1, dec_logvar1, z, enc_logvar1=vae_algorithm.model.forward(img1)
        dec_mean2, dec_logvar2, z2, enc_logvar2=vae_algorithm.model.forward(img2)

        for i in range(z.size()[0]):
            z_np=z[i,:].cpu().detach().numpy()
            z2_np=z2[i,:].cpu().detach().numpy()
            ac_np=ac[i].cpu().detach().numpy()
            latent_map.append((z_np,z2_np,ac_np))
    
    #dump pickle
    with open(output_file+'.pkl', 'wb') as f:
        pickle.dump(latent_map, f)
