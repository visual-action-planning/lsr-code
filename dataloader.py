#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:15:30 2019

@author: petrapoklukar

"""
from __future__ import print_function
import torch
import torch.utils.data as data
import random
import sys
import pickle


def preprocess_triplet_data(filename):
    with open('datasets/'+filename, 'rb') as f:
        if sys.version_info[0] < 3:
            data_list = pickle.load(f)
        else:
            data_list = pickle.load(f, encoding='latin1')
        #data_list = pickle.load(f)

    random.seed(2610)

    random.shuffle(data_list)

    splitratio = int(len(data_list) * 0.15)
    train_data = data_list[splitratio:]
    test_data = data_list[:splitratio]

    train_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    train_data))
    test_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    test_data))
    with open('datasets/train_'+filename, 'wb') as f:
        pickle.dump(train_data1, f)
    with open('datasets/test_'+filename, 'wb') as f:
        pickle.dump(test_data1, f)


# ----------------------- #
# --- Custom Datasets --- #
# ----------------------- #
class TripletTensorDataset(data.Dataset):
    def __init__(self, dataset_name, split):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
       
        if split == 'test':
            with open('datasets/test_'+self.dataset_name+'.pkl', 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open('datasets/train_'+self.dataset_name+'.pkl', 'rb') as f:
                self.data = pickle.load(f)


    def __getitem__(self, index):
        img1, img2, action = self.data[index]
        return img1, img2, action

    def __len__(self):
        return len(self.data)


class APNDataset(data.Dataset):
    def __init__(self, task_name, dataset_name, split, random_seed, dtype, 
                 img_size):
        self.task_name = task_name
        self.dataset_name =  dataset_name
        self.name = dataset_name + '_' + split
        self.split = split.lower()
        self.random_seed = random_seed
        self.dtype = dtype
        self.img_size = img_size

        # Stacking data
        if self.task_name == 'unity_stacking':
            path = 'action_data/{0}/{1}_{2}_seed{3}.pkl'.format(
                    self.dataset_name, self.dtype, self.split, self.random_seed)

            with open(path, 'rb') as f:
                pickle_data = pickle.load(f)
                self.data = pickle_data['data']
                self.min, self.max = pickle_data['min'], pickle_data['max']

        # Shirt data
        if self.task_name == 'shirt_folding':
            path = './action_data/{0}/{1}_normalised_{2}_seed{3}.pkl'.format(
                    self.dataset_name, self.dtype, self.split, self.random_seed)

            with open(path, 'rb') as f:
                pickle_data = pickle.load(f)
                self.data = pickle_data['data']
                self.min, self.max = pickle_data['min'], pickle_data['max']

    def __getitem__(self, index):
        img1, img2, coords = self.data[index]
        return img1, img2, coords

    def __len__(self):
        return len(self.data)

    