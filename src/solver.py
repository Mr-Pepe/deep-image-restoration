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

        self.lr = optim_args_merged['lr']

        self._reset_histories()




    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.val_loss_history = []
        self.psnr_history = []
        self.num_per_subtask_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_after_iters=1, save_after_epochs=1, start_epoch=0, lr_decay=1):

        optim = self.optim(model.parameters(), **self.optim_args)
        if start_epoch == 0:
            self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        loss_avg = 0

        # Generate save folder
        savepath = '../saves/train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        os.makedirs(savepath)

        n_iters = num_epochs*iter_per_epoch
        i_iter = 0

        num_subtasks = 5
        interpolation_sampling_points = [0, 15, 30, 45, 60, 75]
        num_per_subtask = np.ones((num_subtasks,))*train_loader.batch_size/num_subtasks
        num_per_subtask = num_per_subtask.astype('int')

        if start_epoch != 0:
            num_per_subtask = self.num_per_subtask_history[-1]

        corrupter = Corrupter(interpolation_sampling_points, 'pixel_interpolation')

        print('START TRAIN.')

        # Do the training here
        for i_epoch in range(num_epochs):

            i_epoch += start_epoch

            print("Number of samples per subtask: " + str(num_per_subtask))

            for i_iter_in_epoch, batch in enumerate(train_loader):
                i_iter += 1

                ground_truth, _ = batch

                if len(ground_truth) < train_loader.batch_size:
                    print("Skipped batch")
                    continue


                # Corrupt images here
                corrupted_batch = ground_truth.clone()
                corrupted_batch = corrupter(corrupted_batch, num_per_subtask)


                corrupted_batch = corrupted_batch.to(device)
                ground_truth = ground_truth.to(device)

                out = model(corrupted_batch)

                criterion = torch.nn.MSELoss()
                loss = criterion(out, ground_truth)

                model.zero_grad()
                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.item())
                loss_avg = 99/100*loss_avg + 1/100*loss.item()


                if i_iter%log_after_iters == 0:
                    print("Iteration " + str(i_iter) + "/" + str(n_iters) + "   TRAIN loss: " + str(loss.item()) + "   Avg: " + str(loss_avg))

            if (i_epoch+1)%save_after_epochs == 0:
                model.save(savepath + '/model' + str(i_epoch+1))
                self.save(savepath + '/solver' + str(i_epoch+1))


            # Validate model
            print("Validate model after epoch " + str(i_epoch+1))

            self.num_per_subtask_history.append(num_per_subtask)


            model.to(device)
            psnr = np.zeros(num_subtasks, )
            num_val_batches = 0
            for i, batch in enumerate(val_loader):
                num_val_batches += 1

                num_per_subtask = np.ones((num_subtasks,))*val_loader.batch_size/num_subtasks
                num_per_subtask = num_per_subtask.astype('int')

                ground_truth, _ = batch

                corrupted_batch = ground_truth.clone()
                corrupted_batch = corrupter(corrupted_batch,  num_per_subtask=num_per_subtask)


                corrupted_batch = corrupted_batch.to(device)
                ground_truth = ground_truth.to(device)

                out = model(corrupted_batch)


                #Calculate PSNR
                criterion = torch.nn.MSELoss(reduce=False)
                loss = criterion(out, ground_truth)
                loss = loss.cpu().detach().numpy()
                den = loss.shape[1]*loss.shape[2]*loss.shape[3]

                loss = loss.sum(3).sum(2).sum(1)/den



                index = 0
                for i, num_subtask in enumerate(num_per_subtask):
                    tmp_mse = loss[index:index+num_subtask].sum()/num_subtask

                    psnr[i] += 10*np.log10(1/tmp_mse)

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

            num_per_subtask[:-1] = num_per_subtask[:-1].round()
            num_per_subtask[-1] = train_loader.batch_size-num_per_subtask[:-1].sum()
            num_per_subtask.astype('int')


            # Update learning rate
            self.lr *= lr_decay
            print("Learning rate: " + str(self.lr))
            for i, _ in enumerate(optim.param_groups):
                optim.param_groups[i]['lr'] = self.lr

        model.save(savepath + '/model' + str(i_epoch + 1))
        self.save(savepath + '/solver' + str(i_epoch + 1))

        print('FINISH.')


    def save(self, path):
        print('Saving solver... %s' % path)
        pickle.dump(self, open(path, 'wb'))
