import os
from lib.utils import random_init
from lib.ood_seg import OOD_Model
from lib.configs.parse_arg import opt, args
import lib.method_module as method_module
import lib.lucas.mymethods as mymethods

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    random_init(args.seed)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if args.method is not None:
        opt.model.method = args.method

    method_object = getattr(method_module, opt.model.method)
    method = method_object(opt.model.backbone, opt.model.weight_path)

    ood = OOD_Model(method)
    for name, module in method.named_modules():
        print(name)

    submodule = method.model.module
    submodule = getattr(submodule, 'aspp')
    submodule = getattr(submodule, 'features')
    subfeature = submodule[0][2]
    hook = subfeature.register_forward_hook(mymethods.get_feature_maps)

    run_fn = getattr(ood, args.run)
    run_fn()

    patch_size = 10
    patches_list = mymethods.fold_to_patches(patch_size)

    patches_list_mean, patches_list_std = mymethods.calculate_stats(patches_list)
    #print('patches shape:', [len(patches_list), patches_list[0].shape])
    #print('mean shape:', patches_list_mean[0].shape)

    test_img = plt.imread('./data/final_dataset/road_anomaly/original/animals01_Guiguinto_railway_station_Calves.jpg')

    channels_to_plot = [0, 24, 49, 75, 100, 149, 199, 249]

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))

    axs.flat[0].imshow(test_img)
    axs.flat[0].set_title('Tested Image')

    for ax, channel in zip(axs.flat[1:], channels_to_plot):
        ax.imshow(patches_list_mean[0][channel].cpu().numpy(), cmap='viridis')
        ax.set_title(f'Channel {channel}')
    plt.tight_layout()
    plt.show()

    plt.savefig('./saved_img/test.png', dpi=300)
