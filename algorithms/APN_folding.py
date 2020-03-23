#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:45:20 2020

@author: petrapoklukar
"""

from algorithms import APN_Algorithm
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import cv2
import torch
import algorithms.EarlyStopping as ES


class APN_folding(APN_Algorithm):
    def __init__(self, opt):
        super().__init__(opt)
        self.norm_param_d = None
        
        
    def load_norm_param_d(self, path_to_norm_param_d, random_seed):
        """Loads the dict with normalisation parameters"""
        
        path = path_to_norm_param_d + str(random_seed) + '.pkl'
        with open(path, 'rb') as f:
            self.norm_param_d = pickle.load(f)
        
        self.path_to_norm_param_d = path_to_norm_param_d
        self.norm_mean = np.array([
                self.norm_param_d['inputX_mu'], self.norm_param_d['inputY_mu'], 0, 
                self.norm_param_d['outputX_mu'], self.norm_param_d['outputY_mu']])
        self.norm_std = np.array([
                self.norm_param_d['inputX_std'], self.norm_param_d['inputY_std'], 1, 
                self.norm_param_d['outputX_std'], self.norm_param_d['outputY_std']])
    
    
    def descale_coords(self, x):
        """
        Descales the coordinates from [0, 1] interval back to the original 
        image size.
        """
        rescaled = x * (self.data_max - self.data_min) + self.data_min
        rounded_coords = np.around(rescaled).astype(int)
       
        # Filter out of the range coordinates because MSE can be out
        cropped_rounded_coords = np.maximum(
                self.data_min, np.minimum(rounded_coords, self.data_max))
        
        assert(np.all(cropped_rounded_coords) >= self.data_min)
        assert(np.all(cropped_rounded_coords) <= self.data_max)
        return cropped_rounded_coords.astype(int)
    
    
    def denormalise(self, x):
        """
        Denormalises the coordinates with the mean and std of the training
        split. The resulting coordinates are in the [0, 1] interval.
        """
        denormalised = x.numpy() * self.norm_std + self.norm_mean
        assert(np.all(denormalised) >= 0.)
        assert(np.all(denormalised) <= 1.)
        return denormalised
    
    
    def plot_prediction(self, img1, img2, pred_coords_norm, coords_norm, 
                        split='train', n_subplots=3, new_save_path=None):
        """Plots the APN predictions on the given (no-)action pair."""
        img1 = self.vae.decoder(img1)[0]
        img2 = self.vae.decoder(img2)[0]
        
        # Denormalise & descale coords back to the original size
        pred_coords_denorm = self.denormalise(pred_coords_norm.detach())
        coords_denorm = self.denormalise(coords_norm.detach())
        
        pred_coords = self.descale_coords(pred_coords_denorm)
        coords = self.descale_coords(coords_denorm)

        # If there are outliers
        pad_left = abs(int(self.data_min))
        pad_right = abs(int(self.data_max)) - self.img_size
        
        plt.figure(1)        
        for i in range(n_subplots):
            plt.subplot(n_subplots, 2, 2*i+1)
            
            # Start state predictions and ground truth
            pred_pick_xy = pred_coords[i][:2] + abs(int(self.data_min))
            actual_pick_xy = coords[i][:2] + abs(int(self.data_min))
            state1_img = (img1[i].detach().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
            state1_img_padded = cv2.copyMakeBorder(
                    state1_img, pad_left, pad_right, pad_left, pad_right, cv2.BORDER_CONSTANT)
            marked_img1 = cv2.circle(state1_img_padded, tuple(pred_pick_xy), 10, 
                                     (255, 0, 0), -1)
            marked_img1 = cv2.circle(marked_img1, tuple(actual_pick_xy), 15, 
                                     (0, 255, 0), 4)
            fig=plt.imshow(marked_img1)
            
            # Start state predicted height for the robot and ground truth
            pred_pick_height = round(pred_coords_denorm[i][2])
            actual_pick_height = round(coords_denorm[i][2])
            plt.title('State 1, \nh_pred {0}/h_true {1}'.format(
                    pred_pick_height, actual_pick_height))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            # End state predictions and ground truth
            plt.subplot(n_subplots, 2, 2*i+2)
            pred_place_xy = pred_coords[i][3:] + abs(int(self.data_min))
            actual_place_xy = coords[i][3:] + abs(int(self.data_min))
            state2_img = (img2[i].detach().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
            state2_img_padded = cv2.copyMakeBorder(
                    state2_img, pad_left, pad_right, pad_left, pad_right, cv2.BORDER_CONSTANT)
            marked_img2 = cv2.circle(state2_img_padded, tuple(pred_place_xy), 10, 
                                     (255, 0, 0), -1)
            marked_img2 = cv2.circle(marked_img2, tuple(actual_place_xy), 15, 
                                     (0, 255, 0), 4)
            fig = plt.imshow(marked_img2)
            plt.title('State 2')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
        
        if new_save_path: 
            if 'valid' in split:
                new_save_path += '_pick{0}_place{1}'.format(
                        str(actual_pick_xy), str(actual_place_xy))
            plt.savefig(new_save_path)
        else:
            plt.savefig(self.save_path + '_Predictions' + split + str(self.current_epoch))
        plt.clf()
        plt.close()
        cv2.destroyAllWindows()
        
        
    def train(self, train_dataset, test_dataset, num_workers=0, chpnt_path=''):  
        """Trains an APN model with given the hyperparameters."""    
        dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, 
                num_workers=num_workers, drop_last=False)
        n_data = len(train_dataset)
        self.data_min = train_dataset.min
        self.data_max = train_dataset.max
        self.img_size = train_dataset.img_size
        assert(train_dataset.img_size == test_dataset.img_size)
        
        print(('\nPrinting model specifications...\n' + 
               ' *- Path to the model: {0}\n' + 
               ' *- Training dataset: {1}\n' + 
               ' *- Number of training samples: {2}\n' + 
               ' *- Number of epochs: {3}\n' + 
               ' *- Batch size: {4}\n' 
               ).format(self.model_path, train_dataset.dataset_name, n_data, 
                   self.epochs, self.batch_size))
        
        if chpnt_path: 
            # Pick up the last epochs specs
            self.load_checkpoint(chpnt_path)
    
        else:
            # Initialise the model
            self.model = self.init_model()
            self.start_epoch, self.lr = self.lr_schedule.pop(0)
            try:
                self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
            except:
                self.lr_update_epoch, self.new_lr = self.start_epoch - 1, self.lr
            self.optimiser = self.init_optimiser()
            self.valid_losses = []
            self.epoch_losses = []
            print((' *- Learning rate: {0}\n' + 
                   ' *- Next lr update at {1} to the value {2}\n' + 
                   ' *- Remaining lr schedule: {3}'
                   ).format(self.lr, self.lr_update_epoch, self.new_lr, 
                   self.lr_schedule))            
        
        self.load_vae()
        es = ES.EarlyStopping(patience=20)
        num_parameters = self.count_parameters() 
        self.opt['num_parameters'] = num_parameters
        print(' *- Model parameter/training samples: {0}'.format(
                num_parameters/len(train_dataset)))
        print(' *- Model parameters: {0}'.format(num_parameters))
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                spacing = 1
                print('{0:>2}{1}\n\t of dimension {2}'.format('', name, spacing),  
                      list(param.shape))

        print('\nStarting to train the model...\n' )
        for self.current_epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            self.update_learning_rate(self.optimiser)
            epoch_loss = np.zeros(7)
            
            for batch_idx, (img1, img2, coords) in enumerate(dataloader):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                coords = coords.float().to(self.device)

                # APNet loss
                pred_coords = self.model(img1, img2) 
                (the_loss, inputXloss, inputYloss, inputHloss, outputXloss, 
                 outputYloss) = self.compute_loss(pred_coords, coords) 

                epoch_loss += self.format_loss([
                        the_loss, inputXloss, inputYloss, 
                        inputHloss, outputXloss, outputYloss]) 
    
                # Optimise the model 
                self.optimiser.zero_grad()
                the_loss.backward()
                self.optimiser.step()                

                # Monitoring the learning 
                epoch_loss += self.format_loss([the_loss, inputXloss, inputYloss, 
                                                inputHloss, outputXloss, outputYloss]) 
                
            # Plot learning curves
            epoch_loss /= len(dataloader)
            epoch_loss[-1] = int(self.current_epoch)
            self.epoch_losses.append(epoch_loss)
            self.plot_model_loss()
            
            valid_loss = self.compute_test_loss(test_dataset) 
            valid_loss[-1] = int(self.current_epoch)
            self.valid_losses.append(valid_loss)
            self.plot_learning_curve()
            
            # Update the best model
            try:
                if es.keep_best(valid_loss[0]):
                    self.best_model= {
                            'model': self.model,
                            'epoch': self.current_epoch,
                            'train_loss': epoch_loss[0], 
                            'valid_loss': valid_loss[0]
                        }
                    print(' *- New best model at epoch ', self.current_epoch)
            except AssertionError:
                break

            # Update the checkpoint only if there was no early stopping
            self.save_checkpoint(epoch_loss[0])

            # Print current loss values every epoch    
            if (self.current_epoch + 1) % self.console_print == 0:
                print('Epoch {0}:'.format(self.current_epoch))                
                print('   Train loss: {0:.3f} inputX: {1:.3f} inputY: {2:.3f} inputH: {3:.3f} outputX: {4:.3f} outputY: {5:.3f}'.format(*epoch_loss))
                print('   Valid loss: {0:.3f} inputX: {1:.3f} inputY: {2:.3f} inputH: {3:.3f} outputX: {4:.3f} outputY: {5:.3f}'.format(*valid_loss))
                print('   LR: {0:.6e}\n'.format(self.lr))
            
            # Print validation results when specified
            if (self.current_epoch + 1) % self.snapshot == 0:
                
                # Plot APN predictions
                self.plot_prediction(img1, img2, pred_coords, coords)
                self.model.eval()
    
                # Plot training and validation loss
                self.save_checkpoint(epoch_loss[0], keep=True)

                # Write logs 
                self.save_logs(train_dataset, test_dataset)
                self.plot_snapshot_loss()
                        
        print('Training completed.')
        self.plot_model_loss()
        self.model.eval()
        
        # Save the model
        self.save_checkpoint(epoch_loss[0], keep=True)
        torch.save(self.best_model['model'].state_dict(), self.model_path) 
        
        # Save the best performing model
        best_model_dict = {
                'model': self.best_model['model'].state_dict(), 
                'trained_params': {
                        'data_min': self.data_min,
                        'data_max': self.data_max,
                        'norm_mean': self.norm_mean,
                        'norm_std': self.norm_std
                        }
                }
        torch.save(best_model_dict, self.save_path + '_bestModelAll.pt') 
        
        # Save the last model        
        torch.save(self.model.state_dict(), self.save_path + '_lastModel.pt')
        last_model_dict = {
                'model': self.model.state_dict(),
                'trained_params': {
                        'data_min': self.data_min,
                        'data_max': self.data_max,
                        'norm_mean': self.norm_mean,
                        'norm_std': self.norm_std
                        }
                }
        torch.save(last_model_dict, self.save_path + '_lastModelAll.pt')
        self.save_logs(train_dataset, test_dataset)
        
        # Plot predetermined test images for a fair comparisson among models
        self.plot_test_images(test_dataset)
    
    def score_model(
            self, model_name, path_to_valid_dataset, path_to_result_file, 
            random_seed, load_checkpoint=False, path_to_chpnt=''):
        """Scores a trained model on the test set."""
        
        # Load the data 
        path = path_to_valid_dataset + str(random_seed) + '.pkl'
        with open(path, 'rb') as f:
            valid_data_dict = pickle.load(f)
            threshold_min = valid_data_dict['min']
            threshold_max = valid_data_dict['max']
            valid_data = valid_data_dict['data']

        self.data_min = threshold_min
        self.data_max = threshold_max

        print(' *- Loaded data from: ', path_to_valid_dataset)
        print(' *- Loaded normalisation parameters from: ',  self.path_to_norm_param_d)
        print(' *- Chosen thresholds: ', threshold_min, threshold_max)
        
        # Load the trained vae network
        self.load_vae()
        
        # Load the trained apn network
        self.model = self.init_model()
        if load_checkpoint:
            checkpoint = torch.load(path_to_chpnt, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(' *- APN loaded from a checkpoint.')
        else:
            path_to_trained_APNmodel = './models/{0}/apnet_model.pt'.format(model_name)
            self.model.load_state_dict(torch.load(path_to_trained_APNmodel, 
                                                  map_location=self.device))
        self.model.eval()
        assert(not self.vae.training)
        assert(not self.model.training)

        # Init the counters        
        n_points = 0
        pickX_reg_score, pickY_reg_score, height_reg_score = 0, 0, 0
        placeX_reg_score, placeY_reg_score, total_reg_score = 0, 0, 0
        
        # Compute the error
        for img1, img2, coords_unnorm in valid_data:

            n_points += 1

            # Normalise the targets with the parameters of the training split
            norm_mean = torch.from_numpy(self.norm_mean.squeeze())
            norm_std = torch.from_numpy(self.norm_std.squeeze())
            coords_norm = (coords_unnorm - norm_mean)/norm_std
            coords = coords_norm.unsqueeze(0).float() # add BS 1

            # VAE forward pass
            img1 = img1.to(self.device).unsqueeze(0).float()
            img2 = img2.to(self.device).unsqueeze(0).float()        

            enc_mean1, _ = self.vae.encoder(img1)
            enc_mean2, _ = self.vae.encoder(img2)
            
            # Get the predictions from the APN
            pred_coords = self.model(enc_mean1, enc_mean2).detach() # (1, 5)
            
            # Compute the mse loss and log the resutls
            (the_loss, inputXloss, inputYloss, inputHloss, outputXloss, 
                 outputYloss) = self.compute_loss(pred_coords.float(), coords) 
            
            pickX_reg_score += inputXloss.item()
            pickY_reg_score += inputYloss.item()
            placeX_reg_score += outputXloss.item()
            placeY_reg_score += outputYloss.item()
            total_reg_score += the_loss.item()
            
            # --- Save some cases for visual inspection
            if n_points % 5 == 0:
                self.plot_prediction(
                        enc_mean1, enc_mean2, pred_coords, coords,
                        split='valid' + str(n_points), n_subplots=1)
                        
        results_d = {
                'model_name': model_name,
                'n_points': n_points, 
                'random_seed': self.random_seed,
                'pickXmse': round(pickX_reg_score/n_points, 2), 
                'pickYmse': round(pickY_reg_score/n_points, 2), 
                'heightmse': round(height_reg_score/n_points, 2), 
                'placeXmse': round(placeX_reg_score/n_points, 2),
                'placeYmse': round(placeY_reg_score/n_points, 2),
                'total_score_mse': round(total_reg_score/n_points, 2)
                }
        
        print('\nValidation scores:\n {0}\n'.format(results_d))
        import pandas as pd
        df = pd.DataFrame.from_dict([results_d])
        df.to_csv(path_to_result_file, header=None, index=False, mode='a')
        return results_d
    