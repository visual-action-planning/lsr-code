#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:24:12 2020

@author: petrapoklukar
"""

import action_data.utils_action_data as utils
import pickle
import os, os.path
import torch
from importlib.machinery import SourceFileLoader
import numpy as np
from random import shuffle
import sys
sys.path.append('../architectures/')


def generate_latent_normalised_with_seed(vae_name, apn_name, split, random_seed):
    """
    Normalises the action coordinates and encodes the images using a given VAE.
    
    e.g.
    
    vae_name: VAE config file
    split: 'test'/'train' 
    """
    
    # Set the seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    seed_suffix = str(random_seed)
    print(' *- Random seed set to ', random_seed)
    
    # Load VAE config
    vae_config_file = os.path.join('.', 'configs', vae_name + '.py')
    vae_config = SourceFileLoader(vae_name, vae_config_file).load_module().config 
    
    # Load APN config
    apn_config_file = os.path.join('.', 'configs', apn_name + '.py')
    apn_config = SourceFileLoader(apn_name, apn_config_file).load_module().config 
    
    # Load VAE
    opt = vae_config['vae_opt']
    opt['exp_name'] = vae_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt['device'] = device
    opt['vae_load_checkpoint'] = False
    opt['n_latent_samples'] = 1
    vae = utils.init_vae(opt)
    
    # Load the original data
    data_original_opt_key = 'path_to_original_{0}_data'.format(split)
    original_data = '{0}_seed{1}.pkl'.format(
            apn_config['data_original_opt'][data_original_opt_key], 
            seed_suffix)
    with open(original_data, 'rb') as f:
        action_data_dict = pickle.load(f)
        threshold_min = action_data_dict['min']
        threshold_max = action_data_dict['max']
        action_data = action_data_dict['data']
        
    # Load normalisation parameters of the training split
    training_norm_params = '{0}{1}.pkl'.format(
            apn_config['data_original_opt']['path_to_norm_param_d'], 
            seed_suffix)
    
    with open(training_norm_params, 'rb') as f:
        norm_params = pickle.load(f)
        norm_mean_np = np.array([
                norm_params['inputX_mu'], norm_params['inputY_mu'], 0, 
                norm_params['outputX_mu'], norm_params['outputY_mu']])
        norm_mean = torch.from_numpy(norm_mean_np)
        norm_std_np = np.array([
                norm_params['inputX_std'], norm_params['inputY_std'], 1, 
                norm_params['outputX_std'], norm_params['outputY_std']])
        norm_std = torch.from_numpy(norm_std_np)
        
    print(' *- Loaded data: ', original_data)
    print(' *- Loaded normalisation params: ', training_norm_params)
    print(' *- Chosen thresholds: ', threshold_min, threshold_max)
    print(' *- Len of loaded data: ', len(action_data))
    
    # Apn data preparations    
    generated_data_dir = './action_data/action_{0}'.format(vae_name)
    print(' *- Generated APN data directory: ', generated_data_dir)
    if (not os.path.isdir(generated_data_dir)):
            os.makedirs(generated_data_dir)
        
    # Generate apn data and normalise the action coordinates   
    action_data_latent = []
    i = 0
    for img1, img2, coords in action_data:
        i += 1
        
        # Normalise coordinates
        coords_normalised = (coords - norm_mean)/norm_std
        
        # VAE forward pass: img1 is a torch of shape (3, w, h)
        img1 = img1.to(device).unsqueeze(0).float()
        img2 = img2.to(device).unsqueeze(0).float()
        
        (enc_mean1, z_samples1, dec_mean_original1, 
         dec_mean_samples1) = utils.vae_forward_pass(img1, vae, opt)
        (enc_mean2, z_samples2, dec_mean_original2, 
         dec_mean_samples2) = utils.vae_forward_pass(img2, vae, opt)        
        
        # Save the latent samples and the decodings
        latent_original = [enc_mean1.squeeze().detach().squeeze(), 
                           enc_mean2.squeeze().detach().squeeze(), 
                           coords_normalised]
        action_data_latent.append(latent_original)

        # Enlarge the dataset with a given S
        for j in range(opt['n_latent_samples']):
            latent_sample = [z_samples1[j].detach().squeeze(), 
                             z_samples2[j].detach().squeeze(),
                             coords_normalised]
            action_data_latent.append(latent_sample)
        
        # Plot some of the the reconstructions
        if i % 250 == 0:
            print('     - Processed {0} images.'.format(i))
            utils.plot_decodings(img1, dec_mean_original1, dec_mean_samples1, 
                                 i, 1, generated_data_dir, opt, random_seed)
            utils.plot_decodings(img2, dec_mean_original2, dec_mean_samples2, 
                                 i, 2, generated_data_dir, opt, random_seed)

    # Shuffle the points so the the generated samples are all mixed
    shuffle(action_data_latent)
    print(generated_data_dir)
    print('     - Original dataset size: ', len(action_data))
    print('     - Total samples generated: ', len(action_data_latent))
    
    # Pickle the result
    with open(generated_data_dir + '/latent_normalised_{0}_seed{1}.pkl'.format(split, random_seed), 'wb') as f:
        pickle.dump({'data': action_data_latent, 'min': threshold_min, 
                     'max': threshold_max, 'n_action_pairs': i,
                     'n_original_data': len(action_data),
                     'n_generated_data': len(action_data_latent)}, f)
            

def generate_splits_with_seed(dataset_name, path_to_original_data, random_seed, 
                              threshold_min=0., threshold_max=256., clean_data=True):
    """
    Loads the clean data, rescales the action coordinates to [0, 1], and 
    generates new train/validation/test splits for the given random seed. 
    Generates a dictionary with normalisation parameters for the newly generated
    training split.
    
    e.g.
    
    dataset_name: action_shirt_teleop_20191217_20200109_no_unf_with_size
    """
    
    # Set the seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    print(' *- Random seed set to ', random_seed)
    
    original_data = path_to_original_data + dataset_name + '.pkl'
    if clean_data or check_for_anomalies(original_data):
        
        # Load the clean data
        with open(original_data, 'rb') as f:
            action_data = pickle.load(f)
        
        new_action_data = []
        
        # Scale the action coordinates
        for item in action_data:            
            #(img1, img2, 1, input_uv, output_uv, height)
            img1_scaled = item[0]/255
            img1 = torch.from_numpy(img1_scaled.transpose(2, 0, 1))
            img2_scaled = item[0]/255
            img2 = torch.from_numpy(img2_scaled.transpose(2, 0, 1))
            input_uv = item[3].reshape(-1, 1)        
            input_h = np.array(int(item[5])).reshape(-1, 1)
            output_uv = item[4].reshape(-1, 1)
            coords_array = np.concatenate([input_uv, input_h, output_uv]).astype('float32')
                
            # Scale the rest
            coords_array_scaled = (coords_array - threshold_min)/(threshold_max - threshold_min)
            coords_array_scaled[2] = float(item[5]) # don't rescale the height
            assert(np.all(coords_array_scaled) >= 0. and np.all(coords_array_scaled) <= 1.)
            coords = torch.from_numpy(coords_array_scaled).squeeze()  
            
            new_action_data.append([img1, img2, coords])
    
        # Create new train/validation/test splits for the given random seed
        from sklearn.model_selection import train_test_split
        train, heldout = train_test_split(new_action_data, test_size=0.2, 
                                          shuffle=True, random_state=random_seed)
        test, valid = train_test_split(heldout, test_size=0.65, 
                                       shuffle=True, random_state=random_seed)
        print('     - Len training split: ', len(train))
        print('     - Len test split: ', len(test))
        print('     - Len validation split: ', len(valid))
        
        if (not os.path.isdir(path_to_original_data)):
            os.makedirs(path_to_original_data)
    
        # Save the new splits
        with open('{0}/train_{2}_seed{1}.pkl'.format(
                path_to_original_data, random_seed, dataset_name), 'wb') as f:
            pickle.dump({'data': train, 'min': threshold_min, 'max': threshold_max}, f)
        with open('{0}/test_{2}_seed{1}.pkl'.format(
                path_to_original_data, random_seed, dataset_name), 'wb') as f:
            pickle.dump({'data': test, 'min': threshold_min,'max': threshold_max}, f) 
        with open('{0}/validation_{2}_seed{1}.pkl'.format(
                path_to_original_data, random_seed, dataset_name), 'wb') as f:
            pickle.dump({'data': valid, 'min': threshold_min, 'max': threshold_max}, f)
            
        # Get the normalisation parameters for this training split
        inputX = []
        inputY = []
        outputX = []
        outputY = []
        
        for item in train:
            # item[2] is a tensor of shape 5
            inputX.append(item[2][0].item())
            inputY.append(item[2][1].item())
            
            outputX.append(item[2][3].item())
            outputY.append(item[2][4].item())
        
        print('Len of the new list ', len(inputX), len(train))

        norm_d = {
                'inputX_mu': np.mean(inputX),
                'inputX_std': np.std(inputX),
                'inputY_mu': np.mean(inputY), 
                'inputY_std': np.std(inputY),
                
                'outputX_mu': np.mean(outputX),
                'outputX_std': np.std(outputX),
                'outputY_mu': np.mean(outputY), 
                'outputY_std': np.std(outputY)
                }
        
        # Save the disctionary in the pickle
        with open('{0}/train_norm_param_seed{1}.pkl'.format(
                path_to_original_data, random_seed), 'wb') as f:
            pickle.dump(norm_d, f)
            
            
def check_for_anomalies(path, threshold_min=0., threshold_max=256.):
    """
    Reads the data and checks for anomalies. This needs to be done only once.
    e.g. 
    path: shirt_teleop/action_shirt_teleop_20191217_20200109_no_unf_with_size.pkl
    """
    
    with open(path, 'rb') as f:
        action_data = pickle.load(f)

    # Checking for anomalies
    same_coords = 0
    noaction_pairs = 0
    outliers = 0
    not_scaled = 0

    #(img1 np (3, 256, 256), img2 np (3, 256, 256), height, input_uv, output_uv)
    for item in action_data:
                
        # Check that img1 is scaled to [0, 1]
        if np.sum(item[0] >= 0) != np.prod(item[0].shape):
            not_scaled += 1
            print('Img1 not normalised')
        
        # Check that img2 is scaled to [0, 1]
        if np.sum(item[1] >= 0) != np.prod(item[1].shape):
            not_scaled += 1
            print('Img2 not normalised')
        
        # Check that action is 1
        if item[2] != 1: 
            noaction_pairs += 1
            print('No-action pair found.')
        
        # Check that all coords are between the desider thresholds
        if (np.all(item[3].astype(int) < threshold_min) or np.all(item[3].astype(int) > threshold_max) or
            np.all(item[4].astype(int) < threshold_min) or np.all(item[4].astype(int) > threshold_max)):
            outliers += 1
            print('Outliers found.')
        
        # Check that all coord pairs really are different
        if np.all(item[3].astype(int) == item[4].astype(int)):
            same_coords += 1
            print('Same coords found.')
        
    print(' *- Outliers: ', outliers)
    print(' *- Same coords: ', same_coords)
    print(' *- No action pairs: ', noaction_pairs)
    print(' *- Not scaled: ', not_scaled)
    return (same_coords + noaction_pairs + outliers + not_scaled) == 0


# ---------------------------------------------------------------------------- #
# ----------------------- Archieved below this point ------------------------- #
# ---------------------------------------------------------------------------- #        
def normalise_latent_pkl(dataset_name, split, norm_param_d_path):
    with open(norm_param_d_path, 'rb') as f:
        norm_param_d = pickle.load(f)
    
    data_path = './action_data/{0}/latent_{1}.pkl'.format(dataset_name, split)
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
        data = data_dict['data']
        
    data_n = []
    sanity_check_counter = 1
    for z1, z2, coords in data:
        inputX_n = (coords[0] - norm_param_d['inputX_mu'])/norm_param_d['inputX_std'] # InputX
        inputY_n = (coords[1] - norm_param_d['inputY_mu'])/norm_param_d['inputY_std'] # InputY
        outputX_n = (coords[3] - norm_param_d['outputX_mu'])/norm_param_d['outputX_std'] # OutputX
        outputY_n = (coords[4] - norm_param_d['outputY_mu'])/norm_param_d['outputY_std']# OutputY
        coords_n = torch.Tensor([inputX_n, inputY_n, coords[2], outputX_n, outputY_n])
        data_n.append([z1, z2, coords_n])
        if sanity_check_counter < 5:
            print(coords_n)
            sanity_check_counter += 1
    
    data_dict['data'] = data_n
    new_pkl_dict = {**data_dict, **norm_param_d}
    data_path_normalised = './action_data/{0}/latent_{1}_normalised.pkl'.format(
            dataset_name, split)
    
#    print(new_pkl_dict)
    with open(data_path_normalised, 'wb') as f:
        pickle.dump(new_pkl_dict, f)
        
    
    
def get_normalisation_params(vae_name):
    data_path = 'action_{0}/latent_train.pkl'.format(vae_name)
    
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
        data = data_dict['data']
        
    inputX = []
    inputY = []
    outputX = []
    outputY = []
    
    for item in data:
        # item[2] is a tensor of shape 5
        inputX.append(item[2][0].item())
        inputY.append(item[2][1].item())
        
        outputX.append(item[2][3].item())
        outputY.append(item[2][4].item())
    
    print('Len of the new list ', len(inputX), len(data))
    
    norm_d = {
            'inputX_mu': np.mean(inputX),
            'inputX_std': np.std(inputX),
            'inputY_mu': np.mean(inputY), 
            'inputY_std': np.std(inputY),
            
            'outputX_mu': np.mean(outputX),
            'outputX_std': np.std(outputX),
            'outputY_mu': np.mean(outputY), 
            'outputY_std': np.std(outputY)
            }
    return norm_d
        
    
    
def normalise(vae_name, split, norm_param_d):
    data_path = 'action_{0}/latent_{1}.pkl'.format(vae_name, split)
    
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
        data = data_dict['data']
        
    data_n = []
    for z1, z2, coords in data:
        inputX_n = (coords[0] - norm_param_d['inputX_mu'])/norm_param_d['inputX_std'] # InputX
        inputY_n = (coords[1] - norm_param_d['inputY_mu'])/norm_param_d['inputY_std'] # InputY
        outputX_n = (coords[3] - norm_param_d['outputX_mu'])/norm_param_d['outputX_std'] # OutputX
        outputY_n = (coords[4] - norm_param_d['outputY_mu'])/norm_param_d['outputY_std']# OutputY
        coords_n = torch.Tensor([inputX_n, inputY_n, coords[2], outputX_n, outputY_n])
        print(coords_n)
        data_n.append([z1, z2, coords_n])
    
    data_dict['data'] = data_n
    new_pkl_dict = {**data_dict, **norm_param_d}
    data_path_normalised = 'action_{0}/latent_{1}_normalised.pkl'.format(vae_name, split)
    
#    print(new_pkl_dict)
    with open(data_path_normalised, 'wb') as f:
        pickle.dump(new_pkl_dict, f)
    





            
            
