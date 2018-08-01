from random import shuffle
import numpy as np
import datetime
import torch
from torch.autograd import Variable
import os
import pickle
from src.utils import Corrupter

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.MSELoss):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self.lr = self.optim_args['lr']

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.val_loss_history = []
        self.psnr_history = []
        self.num_per_subtask_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_after_iters=1, save_after_epochs=1, start_epoch=0,
              lr_decay=1, lr_decay_interval=1, save_path='../saves/train', num_subtasks=5, mode='pixel_interpolation',
              min_max_corruption=[0, 70]):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optim = self.optim(model.parameters(), **self.optim_args)

        if start_epoch == 0:
            self._reset_histories()

        iter_per_epoch = len(train_loader)

        # Exponentially filtered training loss
        loss_avg = 0

        # Generate save folder
        save_path = os.path.join(save_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        os.makedirs(save_path)

        # Calculate the total number of minibatches for the training procedure
        n_iters = num_epochs*iter_per_epoch
        i_iter = 0

        # Divide the range of corruption into the given number of subtasks
        sampling_points = np.linspace(min_max_corruption[0], min_max_corruption[1], num_subtasks+1)


        # Initialize the number of training samples per subtask
        if start_epoch != 0:
            num_per_subtask = self.num_per_subtask_history[-1]
        else:
            if not train_loader.batch_size%num_subtasks == 0:
                raise Exception('Batch size must be dividable by the number of subtasks.')

            num_per_subtask = np.ones((num_subtasks,)) * train_loader.batch_size / num_subtasks
            num_per_subtask = num_per_subtask.astype('int')

        # Initialize the corrupter with the given mode and subtask ranges
        corrupter = Corrupter(mode, sampling_points)

        print('START TRAIN.')

        # Do the training here
        for i_epoch in range(num_epochs):

            i_epoch += start_epoch

            # Set model to train mode
            model.train()

            print("Number of samples per subtask: " + str(num_per_subtask))
            self.num_per_subtask_history.append(num_per_subtask)


            for i_iter_in_epoch, batch in enumerate(train_loader):
                i_iter += 1

                ground_truth, _ = batch

                # If the current minibatch does not have the full number of samples, skip it
                if len(ground_truth) < train_loader.batch_size:
                    print("Skipped batch")
                    continue

                # Corrupt images here
                corrupted_batch = ground_truth.clone()
                corrupted_batch = corrupter(corrupted_batch, num_per_subtask)

                # Transfer batch to GPU if available
                corrupted_batch = corrupted_batch.to(device)
                ground_truth = ground_truth.to(device)

                # Forward pass
                restored_images = model(corrupted_batch)

                # Calculate loss
                criterion = torch.nn.MSELoss()
                loss = criterion(restored_images, ground_truth)

                # Packpropagate and update weights
                model.zero_grad()
                loss.backward()
                optim.step()

                # Save loss to history
                self.train_loss_history.append(loss.item())
                loss_avg = 99/100*loss_avg + 1/100*loss.item()

                if i_iter%log_after_iters == 0:
                    print("Iteration " + str(i_iter) + "/" + str(n_iters) + "   Train loss: " + str(loss.item()) + "   Avg: " + str(loss_avg))

            # Save model and solver
            if (i_epoch+1)%save_after_epochs == 0:
                model.save(save_path + '/model' + str(i_epoch + 1))
                self.save(save_path + '/solver' + str(i_epoch + 1))
                model.to(device)

            # Validate model
            print("Validate model after epoch " + str(i_epoch+1))

            # Set model to evaluation mode
            model.eval()

            psnr = np.zeros(num_subtasks, )
            num_val_batches = 0

            # Use same number of samples per subtask for evaluation
            num_per_subtask = np.ones((num_subtasks,)) * val_loader.batch_size / num_subtasks
            num_per_subtask = num_per_subtask.astype('int')

            for i, batch in enumerate(val_loader):
                num_val_batches += 1

                ground_truth, _ = batch

                corrupted_batch = ground_truth.clone()
                corrupted_batch = corrupter(corrupted_batch,  num_per_subtask=num_per_subtask)

                corrupted_batch = corrupted_batch.to(device)
                ground_truth = ground_truth.to(device)

                restored_images = model(corrupted_batch)

                #Calculate PSNR
                criterion = torch.nn.MSELoss(reduce=False)
                loss = criterion(restored_images, ground_truth)
                loss = loss.cpu().detach().numpy()
                den = loss.shape[1]*loss.shape[2]*loss.shape[3]

                loss = loss.sum(3).sum(2).sum(1)/den


                max_intensity = 1
                index = 0
                for i, num_subtask in enumerate(num_per_subtask):
                    tmp_mse = loss[index:index+num_subtask].sum()/num_subtask

                    psnr[i] += 10*np.log10(max_intensity/tmp_mse)

                    index += num_subtask

                self.psnr_history.append(psnr)

            psnr /= num_val_batches

            print("PSNR: " + str(psnr))

            inverse_psnr = 1/psnr
            inverse_psnr_sum = inverse_psnr.sum()

            num_per_subtask = np.zeros_like(psnr)

            # Refocus learning with respect to PSNR
            for i, _ in enumerate(num_per_subtask):
                num_per_subtask[i] = inverse_psnr[i]/inverse_psnr_sum*train_loader.batch_size

            # Number of samples per sub task is rounded, the remaining samples are assigned to the hardest task
            num_per_subtask[:-1] = num_per_subtask[:-1].round()
            num_per_subtask[-1] = train_loader.batch_size-num_per_subtask[:-1].sum()
            num_per_subtask.astype('int')


            # Update learning rate
            if i_epoch%lr_decay_interval == 0:
                self.lr *= lr_decay
                print("Learning rate: " + str(self.lr))
                for i, _ in enumerate(optim.param_groups):
                    optim.param_groups[i]['lr'] = self.lr

        # Save model and solver after training
        model.save(save_path + '/model' + str(i_epoch + 1))
        self.save(save_path + '/solver' + str(i_epoch + 1))

        print('FINISH.')


    def save(self, path):
        print('Saving solver... %s' % path)
        pickle.dump(self, open(path, 'wb'))
