"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class EncoderDecoder(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):

        super(EncoderDecoder, self).__init__()
        channels, height, width = input_dim

        nef = 64
        n_bottleneck = 100

        self.conv1      = nn.Conv2d(3, nef, kernel_size=4, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(nef)
        self.lrelu1     = nn.LeakyReLU(0.2, True)
        self.conv2      = nn.Conv2d(nef, nef*2, kernel_size=4, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(nef*2)
        self.lrelu2     = nn.LeakyReLU(0.2, True)
        self.conv3      = nn.Conv2d(nef*2, nef*4, kernel_size=4, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(nef*4)
        self.lrelu3     = nn.LeakyReLU(0.2, True)
        self.conv4      = nn.Conv2d(nef * 4, nef * 8, kernel_size=4, stride=2, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(nef * 8)
        self.lrelu4     = nn.LeakyReLU(0.2, True)

        self.bottleneck_conv            = nn.Conv2d(nef * 8, n_bottleneck, kernel_size=4)
        self.bottleneck_conv_batchnorm   = nn.BatchNorm2d(n_bottleneck)
        self.bottleneck_conv_lrelu       = nn.LeakyReLU(0.2, True)
        self.bottleneck_deconv          = nn.ConvTranspose2d(n_bottleneck, nef * 8, kernel_size=4)
        self.bottleneck_deconv_batchnorm = nn.BatchNorm2d(nef*8)
        self.bottleneck_deconv_relu     = nn.ReLU(True)

        self.deconv4        = nn.ConvTranspose2d(nef*8, nef*4, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.batchnorm_out4 = nn.BatchNorm2d(nef*4)
        self.relu4          = nn.ReLU(True)
        self.deconv3        = nn.ConvTranspose2d(nef*4, nef*2, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.batchnorm_out3 = nn.BatchNorm2d(nef*2)
        self.relu3          = nn.ReLU(True)
        self.deconv2        = nn.ConvTranspose2d(nef * 2, nef, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.batchnorm_out2 = nn.BatchNorm2d(nef)
        self.relu2          = nn.ReLU(True)


        self.deconv1    = nn.ConvTranspose2d(nef, 3, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.sigmoid       = nn.Sigmoid()


    def forward(self, input):

        x1 = self.conv1(input)
        x1 = self.batchnorm1(x1)
        x1 = self.lrelu1(x1)

        x2 = self.conv2(x1)
        x2 = self.batchnorm2(x2)
        x2 = self.lrelu2(x2)

        x3 = self.conv3(x2)
        x3 = self.batchnorm3(x3)
        x3 = self.lrelu3(x3)

        x4 = self.conv4(x3)
        x4 = self.batchnorm4(x4)
        x4 = self.lrelu4(x4)

        x_bottleneck = self.bottleneck_conv(x4)
        # x_bottleneck = self.bottleneck_conv_batchnorm(x_bottleneck)
        x_bottleneck = self.bottleneck_conv_lrelu(x_bottleneck)
        x_bottleneck = self.bottleneck_deconv(x_bottleneck)
        x_bottleneck = self.bottleneck_deconv_batchnorm(x_bottleneck)
        x_bottleneck = self.bottleneck_deconv_relu(x_bottleneck)

        y4 = self.deconv4(x_bottleneck)
        y4 = self.batchnorm_out4(y4)
        y4 = self.relu4(y4)

        y3 = y4+x3

        y3 = self.deconv3(y3)
        y3 = self.batchnorm_out3(y3)
        y3 = self.relu3(y3)

        y2 = y3+x2

        y2 = self.deconv2(y2)
        y2 = self.batchnorm_out2(y2)
        y2 = self.relu2(y2)

        y1 = y2+x1

        y1 = self.deconv1(y1)

        out = self.sigmoid(y1)

        return out

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self.cpu(), path)
