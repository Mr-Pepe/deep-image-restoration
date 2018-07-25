import numpy as np
import torch


class Corrupter(object):

    def __init__(self, sampling_points, mode='pixel_interpolation'):
        if mode=='pixel_interpolation':
            self.corrupt = self.delete_pixels

        self.sampling_points = sampling_points

    def __call__(self, batch, num_per_subtask):
        return self.corrupt(batch, num_per_subtask)

    def delete_pixels(self, images, num_per_subtask, prob=None):

        sampling_points = self.sampling_points

        index = 0

        if prob != None:
            mask = torch.rand((images.shape[0], images.shape[2], images.shape[3]))

            mask = mask < prob


        else:
            # Return uncorrupted images if batch is smaller than the sum of samples per subtask
            if images.shape[0] < num_per_subtask.sum():
                return images



            num_per_subtask = num_per_subtask.astype('int')
            mask = torch.rand((images.shape[0], images.shape[2], images.shape[3]))

            for i_subtask, num_subtask in enumerate(num_per_subtask):
                probs = np.random.rand(num_subtask,)*(sampling_points[i_subtask+1]-sampling_points[i_subtask])+sampling_points[i_subtask]
                probs /= 100

                probs = np.repeat(probs, images.shape[2]*images.shape[3]).reshape(num_subtask, images.shape[2], images.shape[3])

                mask[index:index+num_subtask] = mask[index:index+num_subtask] < torch.Tensor(probs)

                index += num_subtask


        images.transpose(0, 1)[:, mask.byte()] = 0
        return images

