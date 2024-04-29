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
                 affine=True, track_running_stats=True, adapt = False, n_patches=[9,16], patch_version = False):
        super(Norm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.adapt = adapt
        self.momentum = momentum
        self.mean, self.var = None, None
        self.n_patches = n_patches
        self.patch_version = patch_version

    def forward(self, input):
        print('forward norm')
        print("momentum: ", self.momentum)
        #print('adapt: ', self.adapt)
        self._check_input_dim(input)

        print('intput dim:', input.size())
        if self.patch_version:
            input_patches = mymethods.fold_to_patches_uniform(input, self.n_patches)
            self.mean, self.var = mymethods.calculate_stats(input_patches)
        else:
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
            #print('exponential_average_factor:', exponential_average_factor)
            print('running mean shape:', self.running_mean.shape)
            print('data mean shape:', self.mean.shape)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                mix_mean = exponential_average_factor * self.mean \
                                    + (1 - exponential_average_factor) * self.running_mean[:,None,None].detach()
                # update running_var with unbiased var
                if n != 1:
                    mix_var = exponential_average_factor * self.var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var[:,None,None].detach()
                else:
                    mix_var = exponential_average_factor * self.var  \
                              + (1 - exponential_average_factor) * self.running_var[:,None,None].detach()
            mean = mix_mean
            var = mix_var

        else: # test
            if self.running_mean is None:
                mean = self.mean
                var = self.var
            else:
                mean = self.running_mean
                var = self.running_var

        if self.patch_version:
            output = (input_patches - mean[None,:,:,:,None,None])/(torch.sqrt(var[None,:,:,:,None,None] + self.eps))
            if self.affine:
                output = output * self.weight[None,:,None,None,None,None] + self.bias[None,:,None,None,None,None]
            output = output.view_as(input)
            print('output shape:',output.shape)
        else:
            output = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
            if self.affine:
                output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return output

class PatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                track_running_stats=True, adapt=False, affine=True, n_patches=[1,1]):
        super(PatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.adapt = adapt
        self.momentum = momentum
        self.n_patches = n_patches
        self.mean, self.var = None, None

    def forward(self, input):
        print('forward PATCHnorm')
        print("momentum patchnorm: ", self.momentum)
        # print('adapt: ', self.adapt)
        self._check_input_dim(input)
        patches_input = mymethods.fold_to_patches_uniform(input, self.n_patches)
        print()
        print('folded to patches')
        print()
        B, C, H, W = input.shape

        if self.mean is not None:
            #print('mean shape avant:', self.mean.shape)
            #print('running mean shape avant:', self.running_mean.shape)
            self.mean = self.mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            self.var = self.var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        else:
            self.mean, self.var = mymethods.calculate_stats(patches_input)
            #print('mean shape apres calcul:', self.mean.shape)
            #print('running mean shape apres calcul:', self.running_mean.shape)

        expanded_running_mean = self.running_mean.view(1, -1, 1, 1, 1, 1).expand_as(self.mean)
        expanded_running_var = self.running_var.view(1, -1, 1, 1, 1, 1).expand_as(self.var)

        # calculate running estimates
        if self.training:
            print("Layer not adapted for training")

        elif self.adapt:
            #print('running_mean size:', self.running_mean.shape)
            exponential_average_factor = self.momentum
            n = input.numel() / input.size(1)
            with torch.no_grad():
                mix_means = exponential_average_factor * self.mean \
                           + (1 - exponential_average_factor) * expanded_running_mean.detach()
                # update running_var with unbiased var
                if n != 1:
                    mix_vars = exponential_average_factor * self.var * n / (n - 1) \
                              + (1 - exponential_average_factor) * expanded_running_var.detach()
                else:
                    mix_vars = exponential_average_factor * self.var \
                              + (1 - exponential_average_factor) * expanded_running_var.detach()
            self.mean = mix_means
            self.var = mix_vars

        #print("Shape of patches_input:", patches_input.shape)
        #print("Shape of self.mean:", self.mean.shape)
        #print("Shape of self.var:", self.var.shape)

        print('training? ', self.training, ' adapt?', self.adapt)
        # Normalize each patch
        normalized_patches = (patches_input - self.mean) / torch.sqrt(self.var + self.eps)

        # Fold patches back to the original input shape
        output = normalized_patches.reshape(B, C, H, W)

        # Apply learnable parameters
        print('weights shape:', self.weights.shape)
        print('biais shape:', self.bias.shape)
        output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        # Remove the redundant dimensions to keep only the mean and var per patch
        self.mean = self.mean.squeeze(-1).squeeze(-1)
        self.var = self.var.squeeze(-1).squeeze(-1)

        print('mean shape at the end of patchnorm2d:', self.mean.shape)

        return output