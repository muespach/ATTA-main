import os
from lib.utils import random_init
from lib.ood_seg import OOD_Model
from lib.configs.parse_arg import opt, args
import lib.method_module as method_module
import lib.lucas.mymethods as mymethods

import numpy as np
import pandas as pd
import functools

if __name__ == '__main__':
    random_init(args.seed)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if args.method is not None:
        opt.model.method = args.method

    print('custom bn', args.custom_bn)

    method_object = getattr(method_module, opt.model.method)
    method = method_object(opt.model.backbone, opt.model.weight_path)

    #print(method)

    ood = OOD_Model(method)

    batchnorm_stats = []
    training_stats = {
        'mean': {},
        'var': {}
    }
    last_conv = None
    skip_connection = False
    patch_div = [18, 32]
    for name, module in method.named_modules():
        #print("Layer: ", name, " Module: ", module)
        print(name)
        if name == 'model':
            module.register_forward_hook(mymethods.get_input)
        if "conv1" in name or "conv2" in name or "conv3" in name:
            last_conv = name
        if "bn" in name and ".0" in name:
            # Hook the input to calculate the patch stats
            training_stats['mean'][name] = module.running_mean.cpu().numpy()
            training_stats['var'][name] = module.running_var.cpu().numpy()
            #module.register_forward_hook(
            #    functools.partial(mymethods.get_patch_stats, layer_name=name, patch_size=patch_div,training_stats=training_stats))
            # Save the training statistics
            #stats = mymethods.save_training_stats(module, name, last_conv)
            #batchnorm_stats.append(stats)

    df_batchnorm_stats = pd.DataFrame(batchnorm_stats)
    df_batchnorm_stats.to_csv('./saved_data/batchnorm_stats.csv', index=False)


    run_fn = getattr(ood, args.run)
    run_fn()

    #mymethods.save_images()
    #mymethods.save_kl_div()




    ######################

    def get_domainshift_prob(self, x, threshold = 50.0, beta = 0.1, epsilon = 1e-8):
        # Perform forward propagation
        self.method.anomaly_score(x)

        # Calculate the aggregated discrepancy
        discrepancy = 0
        for i, layer in enumerate(self.method.model.modules()):
            if isinstance(layer, nn.BatchNorm2d):
                mu_x, var_x = layer.mean, layer.var
                mu, var = layer.running_mean, layer.running_var
                # Calculate KL divergence
                discrepancy = discrepancy + 0.5 * (torch.log((var + epsilon) / (var_x + epsilon)) + (var_x + (mu_x - mu) ** 2) / (
                        var + epsilon) - 1).sum().item()

        # Training Data Stat. (Use function 'save_bn_stats' to obtain for different models).
        if opt.model.backbone == 'WideResNet38':
            train_stat_mean = 825.3230302274227
            train_stat_std = 131.76657988963967
        elif opt.model.backbone == 'ResNet101':
            train_stat_mean = 2428.9796256740888
            train_stat_std = 462.1095033939578

        # Normalize KL Divergence to a probability.
        normalized_kl_divergence_values = (discrepancy - train_stat_mean) / train_stat_std
        momentum = sigmoid(beta * (normalized_kl_divergence_values - threshold))
        return momentum



