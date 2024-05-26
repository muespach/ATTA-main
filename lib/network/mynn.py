"""
Custom Norm wrappers to enable sync BN, regular BN and for weight initialization
"""
from lib.configs.parse_arg import opt, args
import torch
import torch.nn as nn
import lib.lucas.mymethods as mymethods
from lib.configs.parse_arg import args

# def Norm2d(in_channels):
#     """
#     Custom Norm Function to allow flexible switching
#     """
#     return nn.BatchNorm2d(in_channels)

def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)


class Norm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, adapt = False):
        super(Norm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.adapt = adapt
        self.momentum = torch.tensor([momentum])
        self.mean, self.var = None, None

    def forward(self, input):
        #print('forward norm')
        #print("momentum Norm2d: ", self.momentum)
        #print('adapt: ', self.adapt)
        self._check_input_dim(input)

        #print('intput dim:', input.size())
        self.mean, self.var = mymethods.calculate_stats(input)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * self.mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * self.var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
            mean = self.mean
            var = self.var

        elif self.adapt:
            exponential_average_factor = self.momentum
            if type(self.momentum) is not float:
                if self.momentum.dim() > 1:
                    exponential_average_factor = self.momentum.mean().detach()

            #('exponential_average_factor:', exponential_average_factor, 'type:', type(exponential_average_factor))
            #print('running mean shape:', self.running_mean.shape)
            #print('data mean shape:', self.mean.shape)

            print()
            print('running mean:', self.running_mean.detach().sum())
            print('running var:', self.running_var.detach().sum())
            print('layer mean:', self.mean.sum())
            print('layer var:', self.var.sum())
            print('momentum:', self.momentum)

            n = input.numel() / input.size(1)
            with torch.no_grad():
                mix_mean = exponential_average_factor * self.mean \
                                    + (1 - exponential_average_factor) * self.running_mean.detach()
                # update running_var with unbiased var
                if n != 1:
                    mix_var = exponential_average_factor * self.var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var.detach()
                else:
                    mix_var = exponential_average_factor * self.var  \
                              + (1 - exponential_average_factor) * self.running_var.detach()
            mean = mix_mean
            var = mix_var

            print('mean for normalization (summed over channels):', mean.sum())
            print('var for normalization:', var.sum())
            print()

        else: # test
            if self.running_mean is None:
                mean = self.mean
                var = self.var
            else:
                mean = self.running_mean
                var = self.running_var

        output = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return output

class PatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, adapt=False, n_patches=[2, 2], patch_version=False):
        super(PatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.adapt = adapt
        self.momentum = torch.tensor([momentum])
        self.mean, self.var = None, None
        self.n_patches = n_patches
        self.patch_version = patch_version

    def forward(self, input):
        print()
        print('PATCHNORM')
        print()
        #print("momentum: ", self.momentum)
        # print('adapt: ', self.adapt)
        self._check_input_dim(input)

        #print('intput dim:', input.size())

        input_patches = mymethods.fold_to_patches_uniform(input, self.n_patches)
        self.mean, self.var = mymethods.calculate_stats(input_patches)

        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * self.mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * self.var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
            mean = self.mean
            var = self.var

        elif self.adapt:
            exponential_average_factor = self.momentum
            if type(exponential_average_factor) is not float:
                if self.mean.shape[-2:] != exponential_average_factor.shape:
                    exponential_average_factor = exponential_average_factor.view(self.mean.shape[-2:])
            n = input.numel() / input.size(1)

            print()
            print('running mean:', self.running_mean.detach().sum())
            print('running var:', self.running_var.detach().sum())
            print('patch layer original mean:', self.mean.sum(dim=0))
            print('patch layer original var:', self.var.sum(dim=0))
            print('momentum:', self.momentum)

            with torch.no_grad():
                mix_mean = exponential_average_factor * self.mean \
                           + (1 - exponential_average_factor) * self.running_mean[:,None,None].detach()
                # update running_var with unbiased var
                if n != 1:
                    mix_var = exponential_average_factor * self.var * n / (n - 1) \
                              + (1 - exponential_average_factor) * self.running_var[:,None,None].detach()
                else:
                    mix_var = exponential_average_factor * self.var \
                              + (1 - exponential_average_factor) * self.running_var[:,None,None].detach()
            mean = mix_mean
            var = mix_var

            print('mean for normalization:', mean.sum())
            print('var for normalization:', var.sum())
            print()

            #stats_visualizer = [self.momentum, self.running_mean.sum(), self.running_var.sum(), self.mean.sum(), self.var.sum()]
            #print(stats_visualizer)

        else:  # test
            if self.running_mean is None:
                mean = self.mean
                var = self.var
            else:
                mean = self.running_mean
                var = self.running_var

        output = (input_patches - mean[None, :, :, :, None, None]) / (
                  torch.sqrt(var[None, :, :, :, None, None] + self.eps))
        if self.affine:
            output = output * self.weight[None, :, None, None, None, None] + self.bias[None, :, None, None, None, None]
        output = output.view_as(input)
        #print('output shape:', output.shape)
        #print('final self.mean:', self.mean.shape)
        #print('final self.var:', self.var.shape)

        return output
