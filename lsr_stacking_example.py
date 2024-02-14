# functions to produce figures

from __future__ import print_function
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
import lsr_utils as lsr

#plot path 
def make_figure_path_result(f_start,f_goal,graph_name,config_file,checkpoint_file,action_config,action_checkpoint_file,distance_type,image_save_name):

    #load graph 
    with open(graph_name, 'rb') as f:
        G = pickle.load(f)
    #G=nx.read_gpickle(graph_name)

    #load VAE
    vae_config_file = os.path.join('.', 'configs', config_file + '.py')
    vae_directory = os.path.join('.', 'models', checkpoint_file)
    vae_config = SourceFileLoader(config_file, vae_config_file).load_module().config 
    vae_config['exp_name'] = config_file
    vae_config['vae_opt']['exp_dir'] = vae_directory  
    vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
    vae_algorithm.load_checkpoint('models/'+config_file+"/"+checkpoint_file)
    vae_algorithm.model.eval()
    print("loaded VAE")

    #load APN
    ap_config_file = os.path.join('.', 'configs', action_config + '.py')
    ap_directory = os.path.join('.', 'models', action_checkpoint_file)
    ap_config = SourceFileLoader(action_config, ap_config_file).load_module().config 
    ap_config['exp_name'] = action_config
    ap_config['model_opt']['exp_dir'] = ap_directory 
    ap_algorithm = getattr(alg, ap_config['algorithm_type'])(ap_config['model_opt'])
    ap_algorithm.load_checkpoint('models/'+action_config+"/"+action_checkpoint_file)
    ap_algorithm.model.eval()
    print("loaded APN")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #get encoding
    f_start=np.expand_dims(f_start,axis=0)
    f_goal=np.expand_dims(f_goal,axis=0)
    #get recon start and goal and z
    x=torch.from_numpy(f_start)
    x=x.float()
    x=x.permute(0,3,1,2)
    x = Variable(x).to(device)
    x2=torch.from_numpy(f_goal)
    x2=x2.float()
    x2=x2.permute(0,3,1,2)
    x2 = Variable(x2).to(device)
    dec_mean1, dec_logvar1, z, enc_logvar1=vae_algorithm.model.forward(x)
    dec_mean2, dec_logvar2, z2, enc_logvar2=vae_algorithm.model.forward(x2)
    dec_start=dec_mean1[0].detach().permute(1,2,0).cpu().numpy()    
    z_start=z[0].cpu().detach().numpy()
    dec_goal=dec_mean2[0].detach().permute(1,2,0).cpu().numpy()
    z_goalt=z2[0].cpu().detach().numpy()

    #get closes start and goal node from graph
    [c1_close_idx, c2_close_idx] = lsr.get_closest_nodes(G, z_start, z_goalt,distance_type)
    #use graph to find paths
    paths=nx.all_shortest_paths(G, source=c1_close_idx, target=c2_close_idx)
    #go to numpy
    f_start=np.squeeze(f_start)
    f_goal=np.squeeze(f_goal)

    all_paths_img=[]
    all_paths_z=[]
   
    buffer_img_v=np.ones((f_start.shape[0],30,3),np.uint8)
    buffer_img_tiny=np.ones((f_start.shape[0],5,3),np.uint8)
    path_length=0
    for path in paths:
        path_img=[]
        path_img.append(f_start)
        path_img.append(buffer_img_v)
        path_z=[]
        path_length=0
        for l in path:
            path_length+=1
            z_pos=G.nodes[l]['pos']
            
            z_pos = torch.from_numpy(z_pos).float().to(device)
            z_pos = z_pos.unsqueeze(0)
            path_z.append(z_pos)
            
            img_rec,_=vae_algorithm.model.decoder(z_pos)            

            img_rec_cv=img_rec[0].detach().permute(1,2,0).cpu().numpy() 
            path_img.append(img_rec_cv)
            path_img.append(buffer_img_tiny)

        path_img = path_img[:-1]
        path_img.append(buffer_img_v)
        path_img.append(f_goal)

        all_paths_img.append(path_img)
        all_paths_z.append(path_z)

    #debug visual paths:
    combo_img_vp=[]
    for i in range(len(all_paths_img)):
        t_path=all_paths_img[i]
        combo_img_vp.append(np.concatenate([t_path[x] for x in range(len(t_path))],axis=1))


    #lets get the actions!
    all_actions=[]
    for i in range(len(all_paths_z)):
        z_p=all_paths_z[i]
        path_action=[]
        for j in range(len(z_p)-1):
            z1_t=z_p[j]
            z2_t=z_p[j+1]
            action_to=ap_algorithm.model.forward(z1_t,z2_t)
            action=action_to.cpu().detach().numpy()
            action = np.squeeze(action)
            path_action.append(action)
        all_actions.append(path_action)


    #inpainting actions!
    off_x=55
    off_y=80
    len_box=60
    p_color=(1,0,0)
    r_color=(0,1,0)
    for i in range(len(all_actions)):
        p_a=all_actions[i]
        p_i=all_paths_img[i]
        img_idx=2
        for j in range(len(p_a)):
            a=p_a[j]
            t_img=p_i[img_idx]

            a=np.round(a*2,0).astype("int")
            px=off_x+a[0]*len_box
            py=off_y+a[1]*len_box
            cv2.circle(t_img, (px,py), 12, p_color, 4)
            rx=off_x+a[3]*len_box
            ry=off_y+a[4]*len_box            
            cv2.circle(t_img, (rx,ry), 8, r_color, -1)
            all_paths_img[i][img_idx]=t_img
            img_idx+=2

    #make to 255
    for i in range(len(all_paths_img)):
        p_i=all_paths_img[i]
        for j in range(len(p_i)):
            t_img=p_i[j]
            t_img=t_img*255
            t_img_f=t_img.astype("uint8").copy()
            all_paths_img[i][j]=t_img_f

    combo_img_vp=[]
    for i in range(len(all_paths_img)):
        t_path=all_paths_img[i]
        combo_img_vp.append(np.concatenate([t_path[x] for x in range(len(t_path))],axis=1))
        buffer_img_h=np.ones((30,combo_img_vp[0].shape[1],3),np.uint8)
        combo_img_vp.append(buffer_img_h)

    print("generated " + str(len(all_paths_img))+ " paths.")
    cv2.imwrite(image_save_name,np.concatenate([combo_img_vp[x] for x in range(len(combo_img_vp)-1)],axis=0))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lable_ls' , type=bool, default=False, help='Lable latent space')
    parser.add_argument('--build_lsr' , type=bool, default=False, help='Build Latent Space Roadmap')
    parser.add_argument('--example' , type=bool, default=False, help='Make example path')
    parser.add_argument('--seed', type=int, required=True, default=999, 
                    help='random seed')
    args = parser.parse_args()

    #Example for Latent Space ROadmap on stacking Task
    rng=int(args.seed)
    distance_type=1
    weight=1.0
    config_file="VAE_UnityStacking_L1"    
    checkpoint_file="vae_lastCheckpoint.pth"
    output_file="labeled_latent_spaces/"+config_file+"_latent_space_map"
    dataset_name="unity_stacking"
    testset_name="evaluation_unity_stacking_graph_classes"
    graph_name=config_file  +"_graph"
    action_config="APN_UnityStacking_L1"
    action_checkpoint_file="apnet_lastCheckpoint.pth"
    image_save_name="stacking_example_"+str(rng).zfill(5)+".png"

    
    #lable latent space
    if args.lable_ls:
        print("labeling latent space")
        lsr.lable_latent_space(config_file,checkpoint_file,output_file,dataset_name)
     

    #bulid graph
    if args.build_lsr:
        print("Building LSR")
        latent_map_file=output_file+'.pkl'
        mean_dist_no_ac, std_dist_no_ac, dist_list = lsr.compute_mean_and_std_dev(latent_map_file, distance_type,action_mode=0) 
        epsilon=mean_dist_no_ac+weight*std_dist_no_ac  
        lsr.build_lsr(latent_map_file,epsilon,distance_type,graph_name, config_file,checkpoint_file,min_edge_w=1,min_node_m=1, directed_graph = False,  hasclasses=False, save_node_imgs=False, verbose = False, save_graph = True)

    #select random start and goal state from training set
    #bulid graph
    if args.example:
        print("Generating example")
        f = open('datasets/'+testset_name+'.pkl', 'rb')
        dataset = pickle.load(f)    
        random.seed(rng)
        start_idx=random.randint(0,len(dataset))
        goal_idx=random.randint(0,len(dataset))
        i_start=dataset[start_idx][0]
        i_goal=dataset[goal_idx][1]   

        make_figure_path_result(i_start/255.,i_goal/255.,'graphs/'+graph_name+'.pkl',config_file,checkpoint_file,action_config,action_checkpoint_file,distance_type,image_save_name)

    print("--finished--")




if __name__== "__main__":
  main()
