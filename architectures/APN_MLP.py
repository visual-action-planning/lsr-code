#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:20:51 2020

@author: petrapoklukar
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class APNet(nn.Module):
    """
    Action proposal network with Linear layers.
    
    Input: latent sample from a VAE of dim latent_dim.
    Output: input_uv, input_h, output_uv
    """
    def __init__(self, opt, trained_params=None):
        super().__init__()
        self.opt = opt
        self.shared_imgnet_dims = opt['shared_imgnet_dims'] 
        self.separated_imgnet_dims = opt['separated_imgnet_dims'] 
        self.shared_repr_dims = opt['shared_reprnet_dims'] 
            
        self.device = opt['device']
        self.dropout = opt['dropout']
        
        # After training parameters for getting the right coords
        if trained_params is not None:
            self.data_max = trained_params['data_max']
            self.data_min = trained_params['data_min']
            self.norm_mean = trained_params['norm_mean']
            self.norm_std = trained_params['norm_std']
            
        
        # --- Shared input img & output img network
        self.shared_imgnet = nn.Sequential()
        for i in range(len(self.shared_imgnet_dims) - 1):
        
            self.shared_imgnet.add_module('shared_imgnet_lin' + str(i), nn.Linear(
                    self.shared_imgnet_dims[i], self.shared_imgnet_dims[i+1]))    
            self.shared_imgnet.add_module('shared_imgnet_relu' + str(i), nn.ReLU())
        
        # --- Separate input img & output img network
        self.input_net = nn.Sequential()
        self.output_net = nn.Sequential()
        
        for i in range(len(self.separated_imgnet_dims) - 1):
        
            self.input_net.add_module('input_net_lin' + str(i), nn.Linear(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1]))    
            self.input_net.add_module('input_net_relu' + str(i), nn.ReLU())
            
            self.output_net.add_module('output_net_lin' + str(i), nn.Linear(
                    self.separated_imgnet_dims[i], self.separated_imgnet_dims[i+1]))    
            self.output_net.add_module('output_net_relu' + str(i), nn.ReLU())    
        
        # --- Shared network
        self.shared_reprnet = nn.Sequential()
        for i in range(len(self.shared_repr_dims) - 1):  
            self.shared_reprnet.add_module('shared_reprnet_lin' + str(i), nn.Linear(
                    self.shared_repr_dims[i], self.shared_repr_dims[i+1]))
            if i != len(self.shared_repr_dims) - 2:
                self.shared_reprnet.add_module('shared_reprnet_relu' + str(i), nn.ReLU())
        
        self.weight_init()
    
    def weight_init(self):
        """
        Weight initialiser.
        """
        initializer = globals()[self.opt['weight_init']]

        for block in self._modules:
            b = self._modules[block]
            if isinstance(b, nn.Sequential):
                for m in b:
                    initializer(m)
            else:
                initializer(b)
                
    def forward(self, img1, img2):
        img1_interrepr = self.shared_imgnet(img1)
        img1_repr = self.input_net(img1_interrepr)
        
        img2_interrepr = self.shared_imgnet(img2)
        img2_repr = self.output_net(img2_interrepr)

        concat_repr = torch.cat([img1_repr, img2_repr], dim=-1)        
        out = self.shared_reprnet(concat_repr)
        return out
    
    def descale_coords(self, x):
        rescaled = x * (self.data_max - self.data_min) + self.data_min
        rounded_coords = np.around(rescaled).astype(int)
        # Filter out of the range coordinates
        cropped_rounded_coords = np.maximum(self.data_min, np.minimum(rounded_coords, self.data_max))
        assert(np.all(cropped_rounded_coords) >= self.data_min)
        assert(np.all(cropped_rounded_coords) <= self.data_max)
        return cropped_rounded_coords.astype(int)
    
    def denormalise(self, x):
        denormalised = x.detach().cpu().numpy() * self.norm_std + self.norm_mean
        assert(np.all(denormalised) >= 0.)
        assert(np.all(denormalised) <= 1.)
        return denormalised
    
    def forward_and_transform(self, img1, img2):
        img1_interrepr = self.shared_imgnet(img1)
        img1_repr = self.input_net(img1_interrepr)
        
        img2_interrepr = self.shared_imgnet(img2)
        img2_repr = self.output_net(img2_interrepr)

        concat_repr = torch.cat([img1_repr, img2_repr], dim=-1)        
        out = self.shared_reprnet(concat_repr)
        
        out_denorm = self.denormalise(out)
        out_denorm_height = out_denorm[:, 2]
        out_descaled = self.descale_coords(out_denorm)
        out_descaled[:, 2] = out_denorm_height
        return out_descaled

# 2 versions of weight initialisation
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.Parameter)):
        m.data.fill_(0)
        print('Param_init')


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.Parameter)):
        m.data.fill_(0)
        print('Param_init')
        
        
def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

