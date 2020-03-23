#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:16:18 2020

@author: petrapoklukar
"""

import matplotlib.pyplot as plt
import torch
import importlib
import sys
sys.path.append('../architectures/')


def init_vae(opt):
    """Loads a pretrained VAE model."""
    path_to_pretrained = './models/{0}/vae_model.pt'.format(opt['exp_name'])
    vae_module = importlib.import_module("architectures.{0}".format(opt['model'])) # just have {0} if running the file directly
    print(' *- Imported module: ', vae_module)
    
    try:
        class_ = getattr(vae_module, opt['model'])
        vae_instance = class_(opt).to(opt['device'])
        print(' *- Loaded {0}.'.format(class_))
    except: 
        raise NotImplementedError(
                'Model {0} not recognized'.format(opt['model']))
    
    if opt['vae_load_checkpoint']:
        checkpoint = torch.load(path_to_pretrained, map_location=opt['device'])
        vae_instance.load_state_dict(checkpoint['model_state_dict'])
        print(' *- Loaded checkpoint.')
    else:
        vae_instance.load_state_dict(torch.load(path_to_pretrained, map_location=opt['device']))
    vae_instance.eval()
    assert(not vae_instance.training)
    return vae_instance


def plot_decodings(img, dec_mean_original, dec_mean_samples, item, imgnum, 
                   path, opt, random_seed=''):
    """Plots original and decoded images."""
    plt.figure(1)
    plt.clf()
    n_plots = opt['n_latent_samples'] + 2
    for i in range(n_plots - 2):
        plt.subplot(2, 2, i+3)
        fig=plt.imshow(dec_mean_samples[i].squeeze().numpy().transpose(1, 2, 0))
        plt.title('decoded samples')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.subplot(2, 2, 2)
    fig=plt.imshow(dec_mean_original.squeeze().numpy().transpose(1, 2, 0)) 
    plt.title('decoded original')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    
    plt.subplot(2, 2, 1)
    fig=plt.imshow(img.squeeze().numpy().transpose(1, 2, 0)) 
    plt.title('original')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(path + '/decodings' + str(item) + str(imgnum) + '_seed' + str(random_seed))
    plt.close()
    
    
def plot_hists(path, train_coords, outlier_min, outlier_max, bins=20):
    """Plots histograms of the input data coordinates."""
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.hist(train_coords[:, 0], bins=bins)
    plt.title('input x')

    plt.subplot(2, 2, 2)
    plt.hist(train_coords[:, 1], bins=bins)
    plt.title('input y')

    plt.subplot(2, 2, 3)
    plt.hist(train_coords[:, 3], bins=bins)
    plt.title('output x')

    plt.subplot(2, 2, 4)
    plt.hist(train_coords[:, 4], bins=bins)
    plt.title('output y')    
    plt.savefig(path + '/coord_hist_' + str(bins) + 'bins')
    plt.show()
    
    plt.figure(2)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.hist(train_coords[train_coords < outlier_min], bins=20)
    plt.title('Coords < ', outlier_min)

    plt.subplot(1, 2, 2)
    plt.hist(train_coords[train_coords > outlier_max], bins=20)
    plt.title('Coords > ', outlier_max)
    plt.savefig(path + '/coord_outofdims_hist_' + str(20) + 'bins')
    plt.show()
    
    
def vae_forward_pass(img, vae, opt):
    """Returns latent samples from a trained VAE."""
    enc_mean, enc_logvar = vae.encoder(img)
    enc_std = torch.exp(0.5*enc_logvar)
    latent_normal = torch.distributions.normal.Normal(enc_mean, enc_std)
    
    z_samples = latent_normal.sample((opt['n_latent_samples'], ))
    if opt['n_latent_samples'] > 1:
        z_samples = z_samples.squeeze()
    
    dec_mean_samples, _ = vae.decoder(z_samples)
    dec_mean_samples = dec_mean_samples.detach()
    
    dec_mean_original, _ = vae.decoder(enc_mean)
    dec_mean_original = dec_mean_original.detach()
    return enc_mean, z_samples, dec_mean_original, dec_mean_samples
    
