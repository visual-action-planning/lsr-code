#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:44:46 2020

@author: petrapoklukar
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import sys
sys.path.insert(0,'..')
sys.path.append('../architectures/')
import importlib
import os
from importlib.machinery import SourceFileLoader
import torch.utils.data as data

# ---
# ====================== Training functions ====================== #
# ---
class APN_Algorithm():
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = opt['batch_size']
        self.epochs = opt['epochs']
        self.snapshot = self.opt['snapshot']
        self.console_print = self.opt['console_print']
        
        self.lr_schedule = opt['lr_schedule']
        self.init_lr_schedule = opt['lr_schedule']
        
        self.current_epoch = None
        self.model = None
        self.optimiser = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt['device'] = self.device
        print(' *- Chosen device: ', self.device)
        
        print(' *- Random seed: ', opt['random_seed'])
        self.random_seed = opt['random_seed']
        torch.manual_seed(opt['random_seed'])
        np.random.seed(opt['random_seed'])
        if self.device == 'cuda': torch.cuda.manual_seed(opt['random_seed'])
        self.save_path = self.opt['exp_dir'] + '/' + self.opt['filename']
        self.model_path = self.save_path + '_model.pt'
    
        self.best_model= {
                'model': self.model,
                'epoch': self.current_epoch,
                'train_loss': None, 
                'valid_loss': None
                }

    
    def count_parameters(self):
        """Counts the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    
    def plot_snapshot_loss(self):
        """Plots epochs vs model loss for a given range of epochs."""
        plt_data = np.stack(self.epoch_losses)
        plt_labels = ['loss', 'inputX', 'inputY', 'inputH', 'outputX', 'outputY']
        for i in range(len(plt_labels)):
            plt.subplot(len(plt_labels),1,i+1)
            plt.plot(np.arange(self.snapshot)+(self.current_epoch//self.snapshot)*self.snapshot,
                     plt_data[self.current_epoch-self.snapshot+1:self.current_epoch+1, i], 
                     label=plt_labels[i])
            plt.ylabel(plt_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.save_path + '_SnapshotLosses_{0}'.format(self.current_epoch))
        plt.clf()
        plt.close()
    
    
    def plot_model_loss(self):
        """Plots epochs vs model loss."""
        # All losses
        plt_data = np.stack(self.epoch_losses)
        plt_labels = ['loss', 'inputX', 'inputY', 'inputH', 'outputX', 'outputY']
        for i in range(len(plt_labels)):
            plt.subplot(len(plt_labels),1,i+1)
            plt.plot(np.arange(self.current_epoch+1),
                     plt_data[:, i], 
                     label=plt_labels[i])
            plt.ylabel(plt_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.save_path + '_Losses')
        plt.clf()
        plt.close()
        
        # Losses on the input coordinates
        fig, ax = plt.subplots()
        ax.plot(plt_data[:, 1], 'g-', linewidth=2, label='inputX loss')
        ax.plot(plt_data[:, 2], 'r-', linewidth=2, label='inputY loss')
        ax.plot()
        ax.legend()
        ax.set_xlim(0, self.epochs)
        ax.set(xlabel='# epochs', ylabel='loss', title='Input Coordinate loss')
        plt.savefig(self.save_path + '_InputCoordLoss')
        plt.close()
        
        # Losses on the output coordinates
        fig, ax = plt.subplots()
        ax.plot(plt_data[:, 4], 'g-', linewidth=2, label='outputX loss')
        ax.plot(plt_data[:, 5], 'r-', linewidth=2, label='outputY loss')
        ax.plot()
        ax.legend()
        ax.set_xlim(0, self.epochs)
        ax.set(xlabel='# epochs', ylabel='loss', title='Output Coordinate loss')
        plt.savefig(self.save_path + '_OutputCoordLoss')
        plt.close()
        
        # Total model loss
        fig2, ax2 = plt.subplots()
        ax2.plot(plt_data[:, 0], 'go-', linewidth=3, label='Model loss')
        ax2.plot()
        ax2.set_xlim(0, self.epochs)
        ax2.set(xlabel='# epochs', ylabel='loss', title='Model loss')
        plt.savefig(self.save_path + '_Loss')
        plt.close()
    
    
    def plot_test_images(self, valid_dataset):
        """Plots sthe APN predictions on a subset of test set."""
        self.model.eval()
        assert(not self.model.training)
        
        batch_size = 5
        valid_batch = torch.utils.data.Subset(valid_dataset, np.arange(0, 100, step=3))
        valid_dataloader = torch.utils.data.DataLoader(
                valid_batch, batch_size, drop_last=True)
        
        for batch_idx, (img1, img2, coords) in enumerate(valid_dataloader):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            coords = coords.float().to(self.device)
            
            # APNet loss
            pred_coords = self.model(img1, img2)                    
            self.plot_prediction(img1, img2, pred_coords, coords, 
                                 split='AfterTraining' + str(batch_idx))
            
    
    def descale_coords(self, x):
        """
        Descales the coordinates from [0, 1] interval back to the original 
        image size.
        
        Defined in each subclass.
        """
        pass

    
    def plot_prediction(self, *args):
        """
        Plots the APN predictions on the given (no-)action pair.
        
        Defined in each subclass.
        """
        pass
        
        
    def plot_learning_curve(self):
        """Plots train and validation learning curves of the APN training."""
        train_losses_np = np.stack(self.epoch_losses)
        valid_losses_np = np.stack(self.valid_losses)
        assert(len(valid_losses_np) == len(train_losses_np))
        
        plt_labels = ['loss', 'inputX', 'inputY', 'inputH', 'outputX', 'outputY']
        for i in range(len(plt_labels)):
            plt.subplot(len(plt_labels),1,i+1)
            plt.plot(train_losses_np[:, i], 'g-', linewidth=2, 
                     label='Train ' + plt_labels[i])
            plt.plot(valid_losses_np[:, i], 'b--', linewidth=2, 
                     label='Valid ' + plt_labels[i])
            plt.ylabel(plt_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.save_path + '_chpntValidTrainLoss')
        plt.clf()
        plt.close()
    

    def compute_loss(self, pred_coords, coords):
        """Computes the loss on the training batch given the criterion."""
        batch_loss = nn.MSELoss(reduction='none')(pred_coords, coords) # (batch, 5)
        per_feat_loss = torch.mean(batch_loss, dim=0) # (5)
        the_loss = torch.sum(per_feat_loss)
        return (the_loss, per_feat_loss[0], per_feat_loss[1], per_feat_loss[2], 
                per_feat_loss[3], per_feat_loss[4])
    
    
    def compute_test_loss(self, valid_dataset):
        """Computes the loss on a test dataset."""
        self.model.eval()
        assert(not self.model.training)
        
        batch_size = min(len(valid_dataset), self.batch_size)
        valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset, batch_size, drop_last=True)
        
        losses = np.zeros(7)
        for batch_idx, (img1, img2, coords) in enumerate(valid_dataloader):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            coords = coords.float().to(self.device)
            
            # APNet loss
            pred_coords = self.model(img1, img2)                
            (the_loss, inputXloss, inputYloss, inputHloss, 
             outputXloss, outputYloss) = self.compute_loss(pred_coords, coords) 

            losses += self.format_loss([the_loss, inputXloss, inputYloss, 
                                        inputHloss, outputXloss, outputYloss]) 
            
        if (self.current_epoch + 1) % self.snapshot == 0:    
            self.plot_prediction(img1, img2, pred_coords, coords, split='test')
            
        n_valid = len(valid_dataloader)
        return losses / n_valid
    
    
    def format_loss(self, losses_list):
        """Rounds the loss and returns an np array for logging."""
        reformatted = list(map(lambda x: round(x.item(), 2), losses_list))
        reformatted.append(int(self.current_epoch))
        return np.array(reformatted)
    
    
    def load_vae(self):
        """Loads a pretrained VAE model for encoding the states."""
        vae_name = self.opt['vae_name']
        path_to_pretrained = 'models/{0}/vae_model.pt'.format(vae_name)
        vae_config_file = os.path.join('configs', vae_name + '.py')
        vae_config = SourceFileLoader(vae_name, vae_config_file).load_module().config 

        vae_opt = vae_config['vae_opt']
        vae_opt['device'] = self.device
        vae_opt['vae_load_checkpoint'] = False
        
        vae_module = importlib.import_module("architectures.{0}".format(vae_opt['model']))
        print(' *- Imported module: ', vae_module)
        
        # Initialise the model
        try:
            class_ = getattr(vae_module, vae_opt['model'])
            vae_instance = class_(vae_opt).to(self.device)
            print(' *- Loaded {0}.'.format(class_))
        except: 
            raise NotImplementedError(
                    'Model {0} not recognized'.format(vae_opt['model']))
        
        # Load the weights
        if vae_opt['vae_load_checkpoint']:
            checkpoint = torch.load(path_to_pretrained, map_location=self.device)
            vae_instance.load_state_dict(checkpoint['model_state_dict'])
            print(' *- Loaded checkpoint.')
        else:
            vae_instance.load_state_dict(torch.load(path_to_pretrained, map_location=self.device))
        vae_instance.eval()
        assert(not vae_instance.training)
        self.vae = vae_instance
    
    
    def init_model(self, trained_params=None):
        """Initialises the APN model."""
        model = importlib.import_module("architectures.{0}".format(self.opt['model_module']))
        print(' *- Imported module: ', model)
        try:
            class_ = getattr(model, self.opt['model_class'])
            instance = class_(self.opt, trained_params=trained_params).to(self.device)
            return instance
        except: 
            raise NotImplementedError(
                    'Model {0} not recognized'.format(self.opt['model_module']))
        
        
    def init_optimiser(self):
        """Initialises the optimiser."""
        print(self.model.parameters())
        if self.opt['optim_type'] == 'Adam':
            print(' *- Initialised Adam optimiser.')
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.opt['optim_type'] == 'RMSprop':
            print(' *- Initialised RMSprop optimiser.')
            return optim.RMSprop(self.model.parameters(), lr=self.lr)
        else: 
            raise NotImplementedError(
                    'Optimiser {0} not recognized'.format(self.opt['optim_type']))
    
    
    def update_learning_rate(self, optimiser):
        """Annealing schedule for learning rates."""
        if self.current_epoch == self.lr_update_epoch:
            for param_group in optimiser.param_groups:
                self.lr = self.new_lr
                param_group['lr'] = self.lr
                print(' *- Learning rate updated - new value:', self.lr)
                try:
                    self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
                except:
                    print(' *- Reached the end of the update schedule.')
                print(' *- Remaining lr schedule:', self.lr_schedule)
                

    def train(self, train_dataset, test_dataset, num_workers=0, chpnt_path=''):  
        """
        Trains an APN model with given the hyperparameters.
        
        Defined in each subclass.
        """    
        pass
        
        
    def score_model(self, *args):
        """
        Scores a trained model on the test set.
        Defined in each subclass.
        """ 
        pass

    
    def save_logs(self, train_dataset, test_dataset):
        """Save training logs."""
        log_filename = self.save_path + '_logs.txt'
        valid_losses = np.stack(self.valid_losses)
        epoch_losses = np.stack(self.epoch_losses)
        
        with open(log_filename, 'w') as f:
            f.write('Model {0}\n\n'.format(self.opt['filename']))
            f.write( str(self.opt) )
            f.writelines(['\n\n', 
                    '*- Model path: {0}\n'.format(self.model_path),
                    '*- Training dataset: {0}\n'.format(train_dataset.dataset_name),
                    '*- Number of training examples: {0}\n'.format(len(train_dataset)),
                    '*- Model parameters/Training examples ratio: {0}\n'.format(
                            self.opt['num_parameters']/len(train_dataset)),
                    '*- Number of testing examples: {0}\n'.format(len(test_dataset)),
                    '*- Learning rate schedule: {0}\n'.format(self.init_lr_schedule),
                    '*- Best model performance at epoch· {0}\n'.format(self.best_model['epoch']),
                    ])
            f.write('*- Train/validation model_loss\n')
            f.writelines(list(map(
                    lambda t, v, e: '{0:>3}Epoch {3:.0f} {1:.3f}/{2:.3f}\n'.format('', t, v, e), 
                    epoch_losses[:, 0], valid_losses[:, 0], epoch_losses[:, -1])))
            
            f.write('*- Train/validation inputX Loss\n')
            f.writelines(list(map(
                    lambda t, v, e: '{0:>3}Epoch {3:.0f} {1:.3f}/{2:.3f}\n'.format('', t, v, e), 
                    epoch_losses[:, 1], valid_losses[:, 1], epoch_losses[:, -1])))
            
            f.write('*- Train/validation inputY Loss\n')
            f.writelines(list(map(
                    lambda t, v, e: '{0:>3}Epoch {3:.0f} {1:.3f}/{2:.3f}\n'.format('', t, v, e), 
                    epoch_losses[:, 2], valid_losses[:, 2], epoch_losses[:, -1])))
            
            f.write('*- Train/validation inputH Loss)\n')
            f.writelines(list(map(
                    lambda t, v, e: '{0:>3}Epoch {3:.0f} {1:.3f}/{2:.3f}\n'.format('', t, v, e), 
                    epoch_losses[:, 3], valid_losses[:, 3], epoch_losses[:, -1])))

            f.write('*- Train/validation outputX Loss)\n')
            f.writelines(list(map(
                    lambda t, v, e: '{0:>3}Epoch {3:.0f} {1:.3f}/{2:.3f}\n'.format('', t, v, e), 
                    epoch_losses[:, 4], valid_losses[:, 4], epoch_losses[:, -1])))
            
            f.write('*- Train/validation outputY Loss)\n')
            f.writelines(list(map(
                    lambda t, v, e: '{0:>3}Epoch {3:.0f} {1:.3f}/{2:.3f}\n'.format('', t, v, e), 
                    epoch_losses[:, 5], valid_losses[:, 5], epoch_losses[:, -1])))
        print(' *- Model saved.\n')
    
    
    def save_checkpoint(self, epoch_ml, keep=False):
        """Saves a checkpoint during the training."""
        if keep:
            path = self.save_path + '_checkpoint{0}.pth'.format(self.current_epoch)
            checkpoint_type = 'epoch'
        else:
            path = self.save_path + '_lastCheckpoint.pth'
            checkpoint_type = 'last'
        training_dict = {
                'last_epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimiser_state_dict': self.optimiser.state_dict(),
                'last_epoch_loss': epoch_ml,
                'valid_losses': self.valid_losses,
                'epoch_losses': self.epoch_losses,
                'snapshot': self.snapshot,
                'console_print': self.console_print,
                'current_lr': self.lr,
                'lr_update_epoch': self.lr_update_epoch, 
                'new_lr': self.new_lr, 
                'lr_schedule': self.lr_schedule
                }
        torch.save({**training_dict, **self.opt}, path)
        print(' *- Saved {1} checkpoint {0}.'.format(self.current_epoch, checkpoint_type))
    
        
    def load_checkpoint(self, path, evalm = False):
        """
        Loads a checkpoint and initialises the models to continue training.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model = self.init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.lr = checkpoint['current_lr']
        self.lr_update_epoch = checkpoint['lr_update_epoch']
        self.new_lr = checkpoint['new_lr']
        self.lr_schedule = checkpoint['lr_schedule']
        self.optimiser= self.init_optimiser()
        self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
                
        self.start_epoch = checkpoint['last_epoch'] + 1
        self.current_epoch = self.start_epoch - 1
        self.snapshot = checkpoint['snapshot']
        self.valid_losses = checkpoint['valid_losses']
        self.epoch_losses = checkpoint['epoch_losses']
        
        self.snapshot = checkpoint['snapshot']
        self.console_print = checkpoint['console_print']
         
        print(('\nCheckpoint loaded.\n' + 
               ' *- Last epoch {0} with loss {1}.\n' 
               ).format(checkpoint['last_epoch'], 
               checkpoint['last_epoch_loss']))
        print(' *- Current lr {0}, next update on epoch {1} to the value {2}'.format(
                self.lr, self.lr_update_epoch, self.new_lr)
              )
        if evalm == False:
            self.model.train()
        else: 
            self.model.eval()
    
    def load_best_model_pkl(self, path):
        """Loads the best performing model."""
        best_model_dict = torch.load(path, map_location=self.device)
        best_model_state_dict = best_model_dict['model']
        best_model_trained_params = best_model_dict['trained_params']
        
        self.model = self.init_model(trained_params=best_model_trained_params)
        self.model.load_state_dict(best_model_state_dict)
        self.model.eval()
    
        
# ---
# ====================== Train the models ====================== #
# ---
if __name__ == '__main__': 
    class APNDataset(data.Dataset):
        def __init__(self, dataset_name, split):
            self.split = split.lower()
            self.dataset_name =  dataset_name
            self.name = self.dataset_name + '_' + self.split
            
            # Toydaya
            if self.dataset_name == 'action_shirt_teleop_BigResNet_md30_wd1_ld128_is256_ipyes':
                
                if split == 'test':                
                    with open('../action_data/action_shirt_teleop_BigResNet_md30_wd1_ld128_is256_ipyes/latent_test.pkl', 'rb') as f:
                        pickle_data = pickle.load(f)
                        self.data = pickle_data['data']
                        self.min, self.max = pickle_data['min'], pickle_data['max']
                else:
                    with open('../action_data/action_shirt_teleop_BigResNet_md30_wd1_ld128_is256_ipyes/latent_train.pkl', 'rb') as f:
                        pickle_data = pickle.load(f)
                        self.data = pickle_data['data']
                        self.min, self.max = pickle_data['min'], pickle_data['max']

            else:
                raise ValueError('Not recognized dataset {0}'.format(self.dataset_name))
        
        def __getitem__(self, index):
            img1, img2, coords = self.data[index]
            return img1, img2, coords
    
        def __len__(self):
            return len(self.data)
     
    train_dataset = APNDataset('action_shirt_teleop_BigResNet_md30_wd1_ld128_is256_ipyes', 'train')
    test_dataset = APNDataset('action_shirt_teleop_BigResNet_md30_wd1_ld128_is256_ipyes', 'test')
    
    
    opt = {
        'filename': 'apnet', 
        'exp_dir': 'APNet_TESTINGTBD',
        'vae_name': 'ShirtTeleop_BigResNet_md30_wd1_ld128_is256_ipyes',
        'model_module': 'APN_Mlp',
        'model_class': 'APNet_Mlp_1Head_2Bodies_1Tail',

        'device': 'cpu',
        'shared_imgnet_dims': [128, 64, 64],
        'separated_imgnet_dims': [64, 32, 16],
        'shared_reprnet_dims': [16*2, 16, 8, 5],             
        'dropout': None,
        'weight_init': 'normal_init',    

        'epochs': 100,
        'batch_size': 32,
        'lr_schedule': [(0, 1e-2), (50, 1e-3)], 
        'snapshot': 10,
        'console_print': 5,    
        'optim_type': 'Adam', 
        'random_seed': 1201
    }

    algorithm = APN_Algorithm(opt)
    algorithm.train(train_dataset, test_dataset)
    
            
