#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:09:37 2020

@author: petrapoklukar
"""

import argparse
import os
from importlib.machinery import SourceFileLoader
import algorithms as alg
from dataloader import APNDataset
import action_data.action_data_stacking as action_data

parser = argparse.ArgumentParser()
parser.add_argument('--exp_apn', type=str, required=True, default='', 
                    help='config file with parameters of the model')
parser.add_argument('--seed', type=int, required=True, default=999, 
                    help='random seed')
parser.add_argument('--chpnt_path', type=str, default='', 
                    help='path to the checkpoint')
parser.add_argument('--num_workers', type=int, default=0,      
                    help='number of data loading workers')
parser.add_argument('--generate_new_splits' , type=int, default=0, 
                    help='generates new splits for apn data generation')
parser.add_argument('--generate_apn_data' , type=int, default=1, 
                    help='generates new apn data')
parser.add_argument('--train_apn' , type=int, default=1, 
                    help='trains the apn network')
parser.add_argument('--eval_apn' , type=int, default=1, 
                    help='evaluates the trained apn network')
parser.add_argument('--cuda' , type=bool, default=False, help='enables cuda')
args_opt = parser.parse_args()


# Load ActionProposalNetwork (APN) config file
apn_exp_name = args_opt.exp_apn + '_seed' + str(args_opt.seed)
apn_config_file = os.path.join('.', 'configs', args_opt.exp_apn + '.py')
apn_directory = os.path.join('.', 'models', apn_exp_name) 
print(apn_directory)
if (not os.path.isdir(apn_directory)):
    os.makedirs(apn_directory)

apn_config = SourceFileLoader(args_opt.exp_apn, apn_config_file).load_module().config 
apn_config['model_opt']['random_seed'] = args_opt.seed
apn_config['model_opt']['exp_name'] = apn_exp_name
apn_config['model_opt']['exp_dir'] = apn_directory # place where logs, models, and other stuff will be stored
print(' *- Loading experiment %s from file: %s' % (args_opt.exp_apn, apn_config_file))
print(' *- Generated logs, snapshots, and model files will be stored on %s' % (apn_directory))

# Generate APN data
if args_opt.generate_new_splits:
    print('\n *- Generating new APN splits for random seed ', args_opt.seed)
    path_to_original_dataset = apn_config['data_original_opt']['path_to_original_dataset']
    original_dataset_name = apn_config['data_original_opt']['original_dataset_name']
    action_data.generate_new_spits(original_dataset_name, path_to_original_dataset, 
                                   args_opt.seed)
    
if args_opt.generate_apn_data:
    print('\n *- Generating APN data....')
    vae_name = apn_config['model_opt']['vae_name'] 
    action_data.generate_data_with_seed(vae_name, args_opt.exp_apn, 'train', args_opt.seed)
    action_data.generate_data_with_seed(vae_name, args_opt.exp_apn, 'test', args_opt.seed)

# Initialise VAE model
algorithm = getattr(alg, apn_config['algorithm_type'])(apn_config['model_opt'])
print(' *- Loaded {0}'.format(apn_config['algorithm_type']))

data_train_opt = apn_config['data_train_opt']
train_dataset = APNDataset(
        task_name=data_train_opt['task_name'],
        dataset_name=data_train_opt['dataset_name'], 
        split=data_train_opt['split'],
        random_seed=args_opt.seed,
        dtype=data_train_opt['dtype'], 
        img_size=data_train_opt['img_size'])

data_test_opt = apn_config['data_test_opt']
test_dataset = APNDataset(
        task_name=data_train_opt['task_name'],
        dataset_name=data_test_opt['dataset_name'], 
        split=data_test_opt['split'],
        random_seed=args_opt.seed,    
        dtype=data_test_opt['dtype'],
        img_size=data_test_opt['img_size'])

assert(test_dataset.dataset_name == train_dataset.dataset_name)
assert(train_dataset.split == 'train')
assert(test_dataset.split == 'test')

if args_opt.num_workers is not None:
    num_workers = args_opt.num_workers    
else:
    num_workers = apn_config_file['apn_opt']['num_workers']

if args_opt.chpnt_path != '':
    args_opt.chpnt_path = apn_directory + args_opt.chpnt_path
    
if args_opt.train_apn:
    algorithm.train(train_dataset, test_dataset, num_workers, args_opt.chpnt_path)

if args_opt.eval_apn:
    results_d = algorithm.score_model( 
            apn_exp_name, 
            apn_config['data_valid_opt']['path_to_data'], 
            apn_config['data_valid_opt']['path_to_result_file'],
            load_checkpoint=False, 
            noise=False)
    
    