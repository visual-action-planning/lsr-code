#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 12:32:19 2020

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

# ---
# ====================== Training functions ====================== #
# ---
class APN_stacking(APN_Algorithm):
    def __init__(self, opt):
        super().__init__(opt)
            

    def descale_coords(self, x):
        """
        Descales the coordinates from [0, 1] interval back to the original 
        image size.
        """
        rescaled = x.numpy() * (self.data_max - self.data_min) + self.data_min
        rounded_coords = np.around(rescaled).astype(int)
        
        # Filter out of the range coordinates because MSE can be out
        cropped_rounded_coords = np.maximum(self.data_min, np.minimum(rounded_coords, self.data_max))
        assert(np.all(cropped_rounded_coords) >= self.data_min)
        assert(np.all(cropped_rounded_coords) <= self.data_max)
        return cropped_rounded_coords.astype(int)
    
    
    def get_box_center_from_x_y(self, array):
        """
        Returns the center coordinates corresponding to the box where the APN
        prediction x and y are pointing to. It assumes top left corner = (0,0), 
        x increasing positively downwards and y towards the right.
        """
        x, y = array[1], array[0]
        cx_vec = [55,115,185] # wrt the image coordinates (x positive towards right)
        cy_vec = [87,140,190] # wrt the image coordinates (y positive towards down)
        cx = cx_vec[y]
        if x == 2:
            cy = 195
        else:
            cy = cy_vec[x]
        return (cx,cy)


    def plot_prediction(self, img1, img2, pred_coords_scaled, coords_scaled, 
                            split='train', n_subplots=3, new_save_path=None):
        """Plots the APN predictions on the given (no-)action pair."""
        img1 = self.vae.decoder(img1)[0]
        img2 = self.vae.decoder(img2)[0]

        # Descale coords back to the original size
        pred_coords = self.descale_coords(pred_coords_scaled.detach())
        coords = self.descale_coords(coords_scaled)
        
        plt.figure(1)        
        for i in range(n_subplots):
            # Start state predictions and ground truth
            plt.subplot(n_subplots, 2, 2*i+1)
            pred_pick_xy = self.get_box_center_from_x_y(pred_coords[i][:2])
            actual_pick_xy = self.get_box_center_from_x_y(coords[i][:2])
            state1_img = (img1[i].detach().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
            marked_img1 = cv2.circle(state1_img, tuple(pred_pick_xy), 10, (255, 0, 0), -1)
            marked_img1 = cv2.circle(marked_img1, tuple(actual_pick_xy), 15, (0, 255, 0), 4)
            fig=plt.imshow(marked_img1)
            
            # Start state predicted height for the robot and ground truth
            pred_pick_height = round(pred_coords_scaled[i][2].detach().item())
            plt.title('State 1, h_pred {0}'.format(pred_pick_height))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            # End state predictions and ground truth
            plt.subplot(n_subplots, 2, 2*i+2)            
            pred_place_xy = self.get_box_center_from_x_y(pred_coords[i][3:])
            actual_place_xy = self.get_box_center_from_x_y(coords[i][3:])
            
            state2_img = (img2[i].detach().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
            marked_img2 = cv2.circle(state2_img, tuple(pred_place_xy), 10, (255, 0, 0), -1)
            marked_img2 = cv2.circle(marked_img2, tuple(actual_place_xy), 15, (0, 255, 0), 4)
            fig=plt.imshow(marked_img2)
            plt.title('State 2')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
        
        if new_save_path: 
            plt.savefig(new_save_path)
        else:
            plt.savefig(self.save_path + '_Predictions' + split + str(self.current_epoch))
        plt.clf()
        plt.close()
        cv2.destroyAllWindows()    


    def train(self, train_dataset, test_dataset, num_workers=0, chpnt_path=''):  
        """Trains an APN model with given hyperparameters."""            
        dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, 
                num_workers=num_workers, drop_last=False)
        n_data = len(train_dataset)
        self.data_min = train_dataset.min
        self.data_max = train_dataset.max
        
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
            self.training_losses = []
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
        torch.save(self.model.state_dict(), self.save_path + '_lastModel.pt')
        self.save_logs(train_dataset, test_dataset)
        
        # Plot predetermined test images for a fair comparisson among models
        self.plot_test_images(test_dataset)
        
    
    def score_model(
            self, model_name, path_to_valid_dataset, path_to_result_file, 
            load_checkpoint=False, path_to_chpnt='', suffix='', noise=False):
        """Scores a trained APN model on the test set."""
        
        # Load the data        
        with open(path_to_valid_dataset, 'rb') as f:
            valid_data = pickle.load(f)
            print(' *- Loaded data from ', path_to_valid_dataset)
            
        if type(valid_data) == dict:
            threshold_min = valid_data['min']
            threshold_max = valid_data['max']
            valid_data = valid_data['data']
        else:            
            threshold_min, threshold_max = 0., 2. 
        self.data_min = threshold_min
        self.data_max = threshold_max
        
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
        pred_coord_list = []
        actual_coord_list = []
        n_points = 0
        
        # Categorical
        pickX_score, pickY_score = 0, 0
        placeX_score, placeY_score = 0, 0
        pick_score, place_score, coord_score, total_score = 0, 0, 0, 0
        
        # MSE
        pickX_reg_score, pickY_reg_score = 0, 0
        placeX_reg_score, placeY_reg_score, total_reg_score = 0, 0, 0

        # Compute the error
        for item in valid_data:
            n_points += 1
            
            # (img1, img2, 1, input_uv, output_uv)
            img1 = item[0]
            img2 = item[1]
            input_uv = item[3].reshape(-1, 1)
            input_h = np.array(1).reshape(-1, 1)
            output_uv = item[4].reshape(-1, 1)
            coords_array = np.concatenate([input_uv, input_h, output_uv]).astype('float32')
            
            # Scale the rest
            coords_array_scaled = (coords_array - threshold_min)/(threshold_max - threshold_min)
            coords_array_scaled[2] = 1.0 # just a dummy value for height
            assert(np.all(coords_array_scaled) >= 0. and np.all(coords_array_scaled) <= 1.)
            
            if noise:
                # Add noise to the coordinates
                tiny_noise = np.random.uniform(-0.1, 0.1, size=(5, 1))
                tiny_noise[2] = 0.
                noisy_coords_array_scaled = coords_array_scaled + tiny_noise
                new_noisy_normalised_coords = np.maximum(0., np.minimum(noisy_coords_array_scaled, 1.))
                coords = torch.from_numpy(new_noisy_normalised_coords).transpose(1, 0) # shape (1, 5)
            else:
                coords = torch.from_numpy(coords_array_scaled).transpose(1, 0) # shape (1, 5)
            
            # VAE forward pass
            img1 = img1.to(self.device).unsqueeze(0).float()
            img2 = img2.to(self.device).unsqueeze(0).float()
            enc_mean1, _ = self.vae.encoder(img1)
            enc_mean2, _ = self.vae.encoder(img2)
            
            # Get the predictions from the APN
            pred_coords = self.model(enc_mean1, enc_mean2)
            pred_coords_np = self.descale_coords(pred_coords.detach()).squeeze()
            coords_np = self.descale_coords(coords).squeeze() # shape (5, )
            
            # Append for histogram plots
            pred_coord_list.append(pred_coords_np.reshape(-1, 1))
            actual_coord_list.append(coords_np.reshape(-1, 1))
            
            # Compute the mse loss and log the resutls
            (the_loss, inputXloss, inputYloss, inputHloss, outputXloss, 
                 outputYloss) = self.compute_loss(pred_coords.float(), coords.float()) 
            
            # MSE loss
            pickX_reg_score += inputXloss.item()
            pickY_reg_score += inputYloss.item()
            placeX_reg_score += outputXloss.item()
            placeY_reg_score += outputYloss.item()
            total_reg_score += the_loss.item()
            
            # Categorical loss on individual coordinates
            correct_pickX = 1 if pred_coords_np[0] == coords_np[0] else 0 
            correct_pickY = 1 if pred_coords_np[1] == coords_np[1] else 0 
            correct_placeX = 1 if pred_coords_np[3] == coords_np[3] else 0
            correct_placeY = 1 if pred_coords_np[4] == coords_np[4] else 0
            
            pickX_score += correct_pickX
            pickY_score += correct_pickY
            placeX_score += correct_placeX
            placeY_score += correct_placeY
            coord_score += (correct_pickX + correct_pickY + correct_placeX + correct_placeY)//4
            
            # Categorical loss on start and end predictions
            correct_pick = 1 if sum(abs(pred_coords_np[:2] - coords_np[:2])) == 0 else 0 
            correct_place = 1 if sum(abs(pred_coords_np[3:] - coords_np[3:])) == 0 else 0
            
            pick_score += correct_pick
            place_score += correct_place
            total_score += (correct_pick + correct_place)//2
            
            # --- Plot some for visual inspection
            if n_points % 100 == 0:
                self.plot_prediction(
                        enc_mean1, enc_mean2, pred_coords, coords,
                        split='valid' + str(n_points), n_subplots=1)
        
        # Normalise the MSE errors
        pickX_reg_score /= n_points
        pickY_reg_score /= n_points
        placeX_reg_score /= n_points
        placeY_reg_score /= n_points
        total_reg_score /= n_points
        
        # Normalise the categorical errors
        pickX_score /= n_points
        pickY_score /= n_points
        placeX_score /= n_points
        placeY_score /= n_points
        coord_score /= n_points
                
        results_d = {
                'model_name': model_name,
                'n_points': n_points, 
                'random_seed': self.random_seed,
                
                'pickX_avgdisterrror': round(pickX_reg_score, 2), 
                'pickY_avgdisterrror': round(pickY_reg_score, 2), 
                'placeX_avgdisterrror': round(placeX_reg_score, 2),
                'placeY_avgdisterrror': round(placeY_reg_score, 2),
                'total_avgdisterrror': round(total_reg_score, 2),
                
                'pickX_score': round(pickX_score, 2),
                'pickY_score': round(pickY_score, 2),
                'placeX_score': round(placeX_score, 2),
                'placeY_score': round(placeY_score, 2),
                'coord_score': round(coord_score, 2),
                'pick_score_per': round(pick_score/n_points, 2), 
                'place_score_per': round(place_score/n_points, 2),
                'total_score_per': round(total_score/n_points, 2),
                }
        
        print('\nValidation scores:')
        print(results_d)
        print('\n')
        import pandas as pd
        df = pd.DataFrame.from_dict([results_d])
        df.to_csv(path_to_result_file, header=None, index=False, mode='a')
        return results_d

