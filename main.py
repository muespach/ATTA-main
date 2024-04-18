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

    method_object = getattr(method_module, opt.model.method)
    method = method_object(opt.model.backbone, opt.model.weight_path)

    ood = OOD_Model(method)

    batchnorm_stats = []
    training_stats = {
        'mean': {},
        'var': {}
    }
    last_conv = None
    skip_connection = False
    patch_size = 10
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
            module.register_forward_hook(
                functools.partial(mymethods.get_patch_stats, layer_name=name, patch_size=patch_size,training_stats=training_stats))
            # Save the training statistics
            stats = mymethods.save_training_stats(module, name, last_conv)
            batchnorm_stats.append(stats)

    df_batchnorm_stats = pd.DataFrame(batchnorm_stats)
    df_batchnorm_stats.to_csv('./saved_data/batchnorm_stats.csv', index=False)


    run_fn = getattr(ood, args.run)
    run_fn()

    mymethods.save_images()
    mymethods.save_kl_div()


