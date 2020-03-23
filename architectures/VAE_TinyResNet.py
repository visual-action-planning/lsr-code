#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:53:52 2019

@author: petrapoklukar
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

class ResBlock_Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, kernel_size=3, dropout=0):
        super(ResBlock_Downsample, self).__init__()
        padding = int((kernel_size-1)/2)
        self.res_layer = nn.Sequential()
        for d in range(depth):
            self.res_layer.add_module('res_bn' + str(d), nn.BatchNorm2d(in_channels))
            self.res_layer.add_module('res_relu' + str(d), nn.ReLU())
            self.res_layer.add_module('P_res_arelu'+str(d), TempPrintShape('Input to  res_conv'))
            self.res_layer.add_module('res_conv'+ str(d), nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride=2, padding=padding))
            if d == 0:
                self.res_layer.add_module('res_dropout'+ str(d), nn.Dropout(p=dropout))
            self.res_layer.add_module('P_res_conv'+str(d),TempPrintShape('Output of res_conv'))

        self.skip_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=4, padding=1)

    def forward(self, feat):
        return self.res_layer(feat) + self.skip_layer(feat)


class ResBlock_Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, kernel_size=3, dropout=0):
        super(ResBlock_Upsample, self).__init__()
        padding = int((kernel_size-1)/2)
        self.res_layer = nn.Sequential()
        for d in range(depth):
            self.res_layer.add_module('res_bn' + str(d), nn.BatchNorm2d(in_channels))
            self.res_layer.add_module('res_relu' + str(d), nn.ReLU())
            self.res_layer.add_module('P_res_arelu'+str(d), TempPrintShape('Input to  res_conv'))
            self.res_layer.add_module('res_conv'+ str(d), nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=2, padding=padding,
                output_padding=1))
            if d == 0:
                self.res_layer.add_module('res_dropout'+ str(d), nn.Dropout(p=dropout))
            self.res_layer.add_module('P_res_conv'+str(d),TempPrintShape('Output of res_conv'))

        self.skip_layer = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=4, padding=padding,
                output_padding=3)

    def forward(self, feat):
        return self.res_layer(feat) + self.skip_layer(feat)


class ScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_per_scale=1, depth_per_block=2,
                 kernel_size=3, dropout=0, keep_dims=False):
        super(ScaleBlock, self).__init__()
        self.scale_layer = nn.Sequential()
        if keep_dims:
            for d in range(block_per_scale):
                self.scale_layer.add_module('scale_' + str(d), ResBlock_Upsample(
                        in_channels, out_channels, depth_per_block, kernel_size, dropout))

        else:
            for d in range(block_per_scale):
                self.scale_layer.add_module('scale_' + str(d), ResBlock_Downsample(
                        in_channels, out_channels, depth_per_block, kernel_size, dropout))

    def forward(self, feat):
        return self.scale_layer(feat)

class FCBlock(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=1, dropout=0):
        super(FCBlock, self).__init__()
        self.fcscale_layer = nn.Sequential()
        for d in range(n_layers):
            self.fcscale_layer.add_module('fcblock_relu' + str(d), nn.ReLU())
            self.fcscale_layer.add_module('P_fcblock_relu'+str(d),TempPrintShape('Input to  fcblock_lin'))
            self.fcscale_layer.add_module('fcblock_lin' + str(d), nn.Linear(in_dim, out_dim))
            self.fcscale_layer.add_module('fcblock_drop' + str(d), nn.Dropout(p=dropout))

    def forward(self, feat):
        return self.fcscale_layer(feat)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Downsample, self).__init__()
        padding = int((kernel_size-1)/2)
        self.downsample_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=2, padding=padding)

    def forward(self, feat):
        return self.downsample_layer(feat)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Upsample, self).__init__()
        self.downsample_layer = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1)

    def forward(self, feat):
        return self.downsample_layer(feat)


class LinToConv(nn.Module):
    def __init__(self, input_dim, n_channels):
        super(LinToConv, self).__init__()
        self.n_channels = n_channels
        self.width = int(np.sqrt((input_dim / n_channels)))

    def forward(self, feat):
        feat = feat.view((feat.shape[0], self.n_channels, self.width, self.width))
        return feat


class ConvToLin(nn.Module):
    def __init__(self):
        super(ConvToLin, self).__init__()

    def forward(self, feat):
        batch, channels, width, height = feat.shape
        feat = feat.view((batch, channels * width * height))
        return feat


class TempPrintShape(nn.Module):
    def __init__(self, message):
        super(TempPrintShape, self).__init__()
        self.message = message

    def forward(self, feat):
#        print(self.message, feat.shape)
        return feat


class VAE_TinyResNet(nn.Module):
    """
    Variational Autoencoder with variation only on encoder,
    convolutional layers and droupout.
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.latent_dim = opt['latent_dim']

        self.device = opt['device']
        self.dropout = opt['dropout']
        self.out_activation = opt['out_activation']

        self.conv1_out_channels = opt['conv1_out_channels']
        self.out_channels = opt['conv1_out_channels']
        self.kernel_size = opt['kernel_size']
        self.num_scale_blocks = opt['num_scale_blocks']
        self.block_per_scale = opt['block_per_scale']
        self.depth_per_block = opt['depth_per_block']
        self.fc_dim = opt['fc_dim']
        self.image_size = opt['image_size']
        self.input_channels = opt['input_channels']
        self.learn_dec_logvar = opt['learn_dec_logvar']
        self.decoder_fn = self.decoder_mean_var if opt['learn_dec_logvar'] else self.decoder_mean

        self.latent_conv1_out_channels = opt['latent_conv1_out_channels']
        self.enc_kernel_list = [5, 5]
        self.enc_out_channel_list = [32, 128]

        #--- Encoder network
        self.enc_conv = nn.Sequential()

        self.enc_conv.add_module('enc_conv0', nn.Conv2d(
                self.input_channels, self.enc_out_channel_list[0], self.enc_kernel_list[0],
                stride=2, padding=int((self.enc_kernel_list[0]-1)/2)))
        self.enc_conv.add_module('P_enc_conv0',TempPrintShape('Output of enc_conv0'))
        for d in range(self.num_scale_blocks):
            kernel_size = self.enc_kernel_list[d]
            out_channels = self.enc_out_channel_list[d]
            self.enc_conv.add_module('enc_scale' + str(d),
                    ScaleBlock(out_channels, out_channels,
                               self.block_per_scale, self.depth_per_block,
                               kernel_size, self.dropout))

            if d != self.num_scale_blocks - 1:
                in_channels = self.out_channels
                self.out_channels = self.enc_out_channel_list[d+1]
                self.enc_conv.add_module('P_enc_bdownscale'+str(d),TempPrintShape('Input to  enc_downscale'))
                self.enc_conv.add_module('enc_downscale' + str(d), Downsample(
                        in_channels, self.out_channels, self.kernel_size))
                self.enc_conv.add_module('P_enc_adownscale'+str(d),TempPrintShape('Output of enc_downscale'))

        self.enc_conv.add_module('P_enc_bpool',TempPrintShape('Input to  enc_avgpool'))
        self.enc_conv.add_module('enc_avgpool', nn.AvgPool2d(2))
        self.enc_conv.add_module('P_enc_apool',TempPrintShape('Input to  enc_flatten'))
        self.enc_conv.add_module('enc_flatten', ConvToLin())
        self.enc_conv.add_module('P_enc_bfcs',TempPrintShape('Input to  enc_fcscale'))
        self.enc_conv.add_module('enc_fcscale', FCBlock(
                self.fc_dim, self.fc_dim, 1, self.dropout))
        self.enc_conv.add_module('P_enc_afcs',TempPrintShape('Output of enc_fcscale'))
        self.enc_mean = nn.Linear(self.fc_dim, self.latent_dim)
        self.enc_logvar = nn.Linear(self.fc_dim, self.latent_dim)

        #--- Decoder network
#        scales, dims = self.get_decoders_shape()
        scales = [2, 4, 16]
        dims = [128, 64, 32]

        self.dec_conv = nn.Sequential()
        self.dec_conv.add_module('dec_lin0', nn.Linear(
                self.latent_dim, self.latent_conv1_out_channels*2*2))
        self.dec_conv.add_module('P_dec_bureshape',TempPrintShape('Input to  dec_reshape'))
        self.dec_conv.add_module('dec_reshape', LinToConv(
                self.latent_conv1_out_channels*2*2, self.latent_conv1_out_channels))
        self.dec_conv.add_module('P_dec_bupsampe',TempPrintShape('Input to  dec_upsample'))

        for d in range(len(scales)-1):
            self.dec_conv.add_module('dec_upsample' + str(d), Upsample(
                    dims[d], dims[d+1], self.kernel_size))
            self.dec_conv.add_module('P_dec_bscale' + str(d),TempPrintShape('Input to  dec_scale'))
            self.dec_conv.add_module('dec_scale' + str(d), ScaleBlock(
                    dims[d+1], dims[d+1], self.block_per_scale, self.depth_per_block,
                    self.kernel_size, self.dropout, keep_dims=True))


        # Output mean
        self.dec_mean = nn.Sequential()
        self.dec_mean.add_module('dec_mean', Upsample(
                dims[-1], self.input_channels, self.kernel_size))
        if opt['out_activation'] == 'sigmoid':
            self.dec_mean.add_module('dec_meanact', nn.Sigmoid())

        # Output var as well
        if self.learn_dec_logvar:
            self.dec_logvar = Upsample(
                dims[-1], self.input_channels, self.kernel_size)
            print(' *- Learned likelihood variance.')
        print(' *- Last layer activation function: ', self.out_activation)

        #--- Weight init
        self.weight_init()

    def get_decoders_shape(self):
        """
        """
        desired_scale = self.image_size
        scales, dims = [], []
        current_scale, current_dim = 2, self.latent_conv1_out_channels # This is new
        while current_scale <= desired_scale:
            scales.append(current_scale)
            dims.append(current_dim)
            current_scale *= 2
            current_dim = min(int(current_dim/2), 1024)
        assert(scales[-1] == desired_scale)
        return scales, dims

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

    def encoder(self, x):
        """
        Encoder forward step. Returns mean and log variance.
        """
        # Input (batch_size, Channels=1, Width=28, Height=28)
        x = self.enc_conv(x) # (batch_size, lin_before_latent_dim)
        mean = self.enc_mean(x) # (batch_size, latent_dim)
        logvar = self.enc_logvar(x) # (batch_size, latent_dim)
        return mean, logvar

    def decoder(self, z):
        """
        Decoder forward step. Points to the correct decoder depending on whether
        or not the variance of the likelihood function is learned or not.
        """
        return self.decoder_mean_var(z) if self.learn_dec_logvar else self.decoder_mean(z)

    def decoder_mean(self, z):
        """
        Decoder forward step. Returns mean. Variance is fixed to 1.
        """
        x1 = self.dec_conv(z)
        mean = self.dec_mean(x1)
        logvar = torch.zeros(mean.shape, device=self.device)
        return mean, logvar

    def decoder_mean_var(self, z):
        """
        Decoder forward step. Returns mean and log variance.
        """
        x1 = self.dec_conv(z)
        print(x1.shape)
        mean = self.dec_mean(x1)
        print(mean.shape)
        logvar = self.dec_logvar(x1)
        return mean, logvar

    def sample(self, mean, logvar, sample=False):
        """
        Samples z from the given mean and logvar.
        """
        if self.training or sample:
            std = torch.exp(0.5*logvar)
            eps = torch.empty(std.size(), device=self.device).normal_()
            return eps.mul(std).add(mean)
        else:
            return mean

    def forward(self, x, sample_latent=False, latent_code=False):
        latent_mean, latent_logvar = self.encoder(x)
        z = self.sample(latent_mean, latent_logvar, sample=sample_latent)

        if latent_code:
            return z.squeeze()
        else:
            out_mean, out_logvar = self.decoder_fn(z)
            return out_mean, out_logvar, latent_mean, latent_logvar


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


def create_model(opt):
    return VAE_TinyResNet(opt)


def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    size = 256
    opt = {
            'device': 'cpu',
            'input_channels': 3,
            'latent_dim': 64,
            'out_activation': 'sigmoid',
            'dropout': 0.2,
            'weight_init': 'normal_init',
            'conv1_out_channels': 32,
            'latent_conv1_out_channels': 128,
            'kernel_size': 3,
            'num_scale_blocks': 2,
            'block_per_scale': 1,
            'depth_per_block': 2,
            'fc_dim': 512,
            'image_size': 256,
            'learn_dec_logvar': True
            }

    net = create_model(opt)
    x = torch.autograd.Variable(torch.FloatTensor(5, opt['input_channels'], size, size).uniform_(-1,1))
    latent_mean, latent_logvar = net.encoder(x)
    z = net.sample(latent_mean, latent_logvar)
    print('\n * ---')
    out = net(x)
    print('\n * ---')
    print(x.shape)
    print(out[-1].shape)
    print(out[0].shape)
