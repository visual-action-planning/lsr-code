#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 08:03:26 2020

@author: petrapoklukar
"""

batch_size = 64

config = {}

# set the parameters related to the training and testing set
data_train_opt = {} 
data_train_opt['batch_size'] = batch_size
data_train_opt['dataset_name'] = 'action_VAE_ShirtFolding_L2'
data_train_opt['split'] = 'train'
data_train_opt['dtype'] = 'latent'
data_train_opt['img_size'] = 256
data_train_opt['task_name'] = 'shirt_folding'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['epoch_size'] = None
data_test_opt['dataset_name'] = 'action_VAE_ShirtFolding_L2'
data_test_opt['split'] = 'test'
data_test_opt['dtype'] = 'latent'
data_test_opt['img_size'] = 256
data_test_opt['task_name'] = 'shirt_folding'

data_valid_opt = {}
data_valid_opt['path_to_data'] =  './action_data/action_shirt_folding/validation_action_shirt_folding_seed'
data_valid_opt['path_to_result_file'] = './models/APN_ShirtFolding_evaluation_results.txt'
data_valid_opt['split'] = 'valid'
data_valid_opt['img_size'] = 256

data_original_opt = {}
data_original_opt['path_to_original_dataset'] = './action_data/action_shirt_folding/'
data_original_opt['original_dataset_name'] = 'action_shirt_folding' 
data_original_opt['path_to_original_train_data'] = './action_data/action_shirt_folding/train_action_shirt_folding'
data_original_opt['path_to_original_test_data'] = './action_data/action_shirt_folding/test_action_shirt_folding'
data_original_opt['path_to_norm_param_d'] = './action_data/action_shirt_folding/train_norm_param_seed'

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt
config['data_valid_opt'] = data_valid_opt
config['data_original_opt'] = data_original_opt

model_opt = {
    'filename': 'apnet', 
    'vae_name': 'VAE_ShirtFolding_L2', # to decode samples with
    'model_module': 'APN_MLP',
    'model_class': 'APNet',

    'device': 'cpu',
    'shared_imgnet_dims': [64, 64],
    'separated_imgnet_dims': [64, 16],
    'shared_reprnet_dims': [16*2, 16, 5],     
    'dropout': None,
    'weight_init': 'normal_init',    

    'epochs': 500,
    'batch_size': batch_size,
    'lr_schedule':  [(0, 1e-2), (100, 5e-3), (200, 1e-3), (300, 1e-4)], 
    'snapshot': 50,
    'console_print': 5,    
    'optim_type': 'Adam', 
    'random_seed': 1201,
    'num_workers': 4,
    'random_seeds': []
}

config['model_opt'] = model_opt
config['algorithm_type'] = 'APN_folding'
