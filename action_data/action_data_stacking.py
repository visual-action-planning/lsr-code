#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:20:04 2020

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


def generate_data_with_seed(vae_name, apn_exp_name, split, random_seed):
    """
    vae_name: VAE config file
    apn_exp_name: APN config file
    split: 'test'/'train'
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    print(' *- Random seed set to ', random_seed)
    
    # Load VAE config
    vae_config_file = os.path.join('.', 'configs', vae_name + '.py')
    vae_config = SourceFileLoader(vae_name, vae_config_file).load_module().config 
    print(' *- VAE loaded from: ', vae_config_file)
    
    # Load APN config
    apn_config_file = os.path.join('.', 'configs', apn_exp_name + '.py')
    apn_config = SourceFileLoader(apn_exp_name, apn_config_file).load_module().config 
    print(' *- APN loaded from: ', apn_config_file)
    
    # Load VAE
    opt = vae_config['vae_opt']
    opt['exp_name'] = vae_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt['device'] = device
    opt['vae_load_checkpoint'] = False
    opt['n_latent_samples'] = 2
    vae = utils.init_vae(opt)
    
    # Load the original data
    data_original_opt_key = 'path_to_original_{0}_data'.format(split)
    original_data_path = apn_config['data_original_opt'][data_original_opt_key]
    original_data_path_with_seed = '{0}_seed{1}.pkl'.format(
            original_data_path.split('.pkl')[0], random_seed)
    print(' *- Loading checked and scaled original data from: ', original_data_path_with_seed)
    
    with open(original_data_path_with_seed, 'rb') as f:
        action_data_dict = pickle.load(f)
        threshold_min = action_data_dict['min']
        threshold_max = action_data_dict['max']
        action_data = action_data_dict['data']
        print(' *- Loaded data with thresholds: ', threshold_min, threshold_max, 
              ' and len ', len(action_data))

    # Apn data preparations    
    generated_data_name = 'action_' + vae_name    
    generated_data_dir = './action_data/' + generated_data_name
    print(' *- Generated APN data directory: ',generated_data_dir )
    if (not os.path.isdir(generated_data_dir)):
            os.makedirs(generated_data_dir)

    action_data_latent = []
    i = 0
    for img1, img2, coords in action_data:
        #(img1, img2, coords)
        i += 1
        
        # VAE forward pass
        img1 = img1.to(device).unsqueeze(0).float()
        img2 = img2.to(device).unsqueeze(0).float()
        
        (enc_mean1, z_samples1, dec_mean_original1, 
         dec_mean_samples1) = utils.vae_forward_pass(img1, vae, opt)
        (enc_mean2, z_samples2, dec_mean_original2, 
         dec_mean_samples2) = utils.vae_forward_pass(img2, vae, opt)
        
        # Save the latent samples and the decodings
        latent_original = [enc_mean1.squeeze().detach(), 
                           enc_mean2.squeeze().detach(), coords]
        action_data_latent.append(latent_original)

        for j in range(opt['n_latent_samples']):
            latent_sample = [z_samples1[j].detach(), z_samples2[j].detach(),
                             coords]
            action_data_latent.append(latent_sample)
        
        
        # Plot the reconstructions
        if i % 900 == 0:
            print('     - Processed {0} images.'.format(i))
            utils.plot_decodings(img1, dec_mean_original1, dec_mean_samples1, i, 1, 
                           generated_data_dir, opt)
            utils.plot_decodings(img2, dec_mean_original2, dec_mean_samples2, i, 2, 
                           generated_data_dir, opt)

    shuffle(action_data_latent)
    print(generated_data_dir)
    print('     - Action_pairs: ', i, '/', len(action_data))
    print('     - Total samples generated: ', len(action_data_latent))
    
    with open(generated_data_dir + '/latent_{0}_seed{1}.pkl'.format(split, random_seed), 
              'wb') as f:
        pickle.dump({'data': action_data_latent, 'min': threshold_min, 
                     'max': threshold_max, 'n_action_pairs': i,
                     'n_original_data': len(action_data),
                     'n_generated_data': len(action_data_latent)}, f)
    
    
def generate_new_spits(dataset_name, path_to_dataset, random_seed, 
                           clean_data=True, noise=False):
    """Reads all the data and regenerates the train/test/validation splits."""
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    print(' *- Random seed set to ', random_seed)
    
    path_to_original_dataset = path_to_dataset + dataset_name + '.pkl'
    if clean_data or check_for_anomalies(path_to_original_dataset):
        with open(path_to_original_dataset, 'rb') as f:
            action_data = pickle.load(f)
            print(' *- Loaded data: ', path_to_original_dataset)
        
        # Stacking data has always the thresholds
        threshold_min, threshold_max = 0., 2.
        print(' *- Thresholds: ', threshold_min, threshold_max)
        print(' *- Noise: ', noise)
        
        scaled_action_data = []

        # (img1 torch (3, 256, 256), img2 torch (3, 256, 256), 
        #  height tensor(1.), input_uv, output_uv)
        for item in action_data:
            input_uv = item[3].reshape(-1, 1)  # torch.shape(2, 1)
            input_h = np.array(1).reshape(-1, 1) 
            output_uv = item[4].reshape(-1, 1)
            coords_array = np.concatenate([input_uv, input_h, output_uv]).astype('float32') # (5, 1)
            
            # Scale the rest
            coords_array_scaled = (coords_array - threshold_min)/(threshold_max - threshold_min) # (5, 1)
            coords_array_scaled[2] = 1.0 # just a dummy value for height 
            assert(np.all(coords_array_scaled) >= 0. and np.all(coords_array_scaled) <= 1.)
            
            # Add noise to the coordinates
            if noise:
                tiny_noise = np.random.uniform(-0.1, 0.1, size=(5, 1))
                tiny_noise[2] = 0.
                noisy_coords_array_scaled = coords_array_scaled + tiny_noise
                new_noisy_normalised_coords = np.maximum(0, np.minimum(noisy_coords_array_scaled, 1))
                coords = torch.from_numpy(new_noisy_normalised_coords).squeeze() # torch.size 5
            else: 
                coords = torch.from_numpy(coords_array_scaled).squeeze()
            scaled_action_data.append([item[0], item[1], coords])
                
        assert(len(action_data) == len(scaled_action_data))
        
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(scaled_action_data, test_size=0.2, 
                                       shuffle=True, random_state=1201)
        print('     - Len training split: ', len(train))
        print('     - Len test split: ', len(test))
        
        if noise:
            path_to_dataset = path_to_dataset + 'noisy_'
       
        new_train_data = 'train_{0}_seed{1}.pkl'.format(dataset_name, random_seed)
        with open(path_to_dataset + new_train_data, 'wb') as f:
            pickle.dump({'data': train, 'min': threshold_min, 'max': threshold_max}, f)
            print(' *- {0} saved.'.format(path_to_dataset + new_train_data))

        new_test_data = 'test_{0}_seed{1}.pkl'.format(dataset_name, random_seed)
        with open(path_to_dataset + new_test_data, 'wb') as f:
            pickle.dump({'data': test, 'min': threshold_min,'max': threshold_max}, f) 
            print(' *- {0} saved.'.format(path_to_dataset + new_test_data))
            
            
def check_for_anomalies(path):
    """Read the stacking data and checks for anomalies."""
    
    with open(path, 'rb') as f:
        action_data = pickle.load(f)
    
    # Stacking data has always these thresholds
    threshold_min, threshold_max = 0., 2.
    
    same_coords = 0
    noaction_pairs = 0
    outliers = 0
    not_scaled = 0
    for item in action_data:
        
        # (img1 torch (3, 256, 256), img2 torch (3, 256, 256), 
        # height tensor(1.), input_uv, output_uv)
        
        # Check that img1 is scaled to [0, 1]
        if torch.sum(item[0] >= 0) != torch.prod(torch.tensor(item[0].shape)):
            not_scaled += 1
            print('Img1 not normalised')
        
        # Check that img2 is scaled to [0, 1]
        if torch.sum(item[1] >= 0) != torch.prod(torch.tensor(item[1].shape)):
            not_scaled += 1
            print('Img2 not normalised')
        
        # Check that action is 1
        if int(item[2]) != 1: 
            noaction_pairs += 1
            print('No-action pair found.')
        
        # Check that all coords are between [0, 2]
        if (torch.all(item[3].int() < threshold_min) or torch.all(item[3].int() > threshold_max) or
            torch.all(item[4].int() < threshold_min) or torch.all(item[4].int() > threshold_max)):
            outliers += 1
            print('Outliers found.')
        
        # Check that all coord pairs really are different
        if torch.all(item[3].int() == item[4].int()):
            same_coords += 1
            print('Same coords found.')
            
    print(' *- Outliers: ', outliers)
    print(' *- Same coords: ', same_coords)
    print(' *- No action pairs: ', noaction_pairs)
    print(' *- Not scaled: ', not_scaled)
    return (same_coords + noaction_pairs + outliers + not_scaled) == 0
            
            