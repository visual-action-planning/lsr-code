#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:58:26 2019

@author: petrapoklukar
"""

from __future__ import print_function
import argparse
import os
from importlib.machinery import SourceFileLoader
import algorithms as alg
from dataloader import TripletTensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--exp_vae', type=str, required=True, default='', 
                    help='config file with parameters of the vae model')
parser.add_argument('--chpnt_path', type=str, default='', 
                    help='path to the checkpoint')
parser.add_argument('--num_workers', type=int, default=0,      
                    help='number of data loading workers')
parser.add_argument('--cuda' , type=bool, default=False, help='enables cuda')
args_opt = parser.parse_args()

# Load VAE config file
vae_config_file = os.path.join('.', 'configs', args_opt.exp_vae + '.py')
vae_directory = os.path.join('.', 'models', args_opt.exp_vae)
if (not os.path.isdir(vae_directory)):
    os.makedirs(vae_directory)

print(' *- Training:')
print('    - VAE: {0}'.format(args_opt.exp_vae))

vae_config = SourceFileLoader(args_opt.exp_vae, vae_config_file).load_module().config 
vae_config['exp_name'] = args_opt.exp_vae
vae_config['vae_opt']['exp_dir'] = vae_directory # the place where logs, models, and other stuff will be stored
print(' *- Loading experiment %s from file: %s' % (args_opt.exp_vae, vae_config_file))
print(' *- Generated logs, snapshots, and model files will be stored on %s' % (vae_directory))

# Initialise VAE model
vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
print(' *- Loaded {0}'.format(vae_config['algorithm_type']))

data_train_opt = vae_config['data_train_opt']
train_dataset = TripletTensorDataset(
    dataset_name=data_train_opt['dataset_name'],
    split=data_train_opt['split'])

data_test_opt = vae_config['data_test_opt']
test_dataset = TripletTensorDataset(
    dataset_name=data_test_opt['dataset_name'],
    split=data_test_opt['split'])
assert(test_dataset.dataset_name == train_dataset.dataset_name)
assert(train_dataset.split == 'train')
assert(test_dataset.split == 'test')

if args_opt.num_workers is not None:
    num_workers = args_opt.num_workers    
else:
    num_workers = vae_config_file['vae_opt']['num_workers']

vae_algorithm.train(train_dataset, test_dataset, num_workers, args_opt.chpnt_path)
