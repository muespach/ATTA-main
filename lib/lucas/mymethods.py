import os
import numpy as np
import pandas as pd

feature_maps = {}
patch_feature_maps = {
    'mean': {},
    'std': {}
}
input_images = []
kl_divs = {}


def get_feature_maps(module, input, output, layer_name):
    global feature_maps
    feature_maps[layer_name] = output.detach().cpu()

def get_patch_stats(module, input, output, layer_name, patch_size, training_stats):
    global patch_feature_maps
    if layer_name not in patch_feature_maps:
        patch_feature_maps['mean'][layer_name] = []
        patch_feature_maps['std'][layer_name] = []
    patch_feature_map = fold_to_patches_uniform(input[0].detach().cpu(), patch_size)
    patch_means, patch_stds = calculate_stats(patch_feature_map)
    patch_feature_maps['mean'][layer_name].append(patch_means[0].detach().cpu().numpy())
    patch_feature_maps['std'][layer_name].append(patch_stds[0].detach().cpu().numpy())
    compare_stats_hook(training_stats, patch_feature_maps, layer_name)
    #print('feature patch type and shape:',type(patch_feature_maps['std'][layer_name][0]),np.shape(patch_feature_maps['std'][layer_name][0]))


def compare_stats_hook(training_stats, feature_maps, bn_layer):
    global kl_divs
    #print(bn_layer)
    # Dynamically reshape based on the first dimension of training_stats['mean'][layer]
    first_dimension_size = training_stats['mean'][bn_layer].shape[0]
    reshaped_train_mean = training_stats['mean'][bn_layer].reshape((first_dimension_size, 1, 1))
    reshaped_train_std = np.sqrt(training_stats['var'][bn_layer]).reshape((first_dimension_size, 1, 1))

    kl_div = kl_divergence(patch_feature_maps['mean'][bn_layer][0], patch_feature_maps['std'][bn_layer][0],
                                  reshaped_train_mean, reshaped_train_std)

    if bn_layer not in kl_divs:
        kl_divs[bn_layer] = []
    kl_divs[bn_layer].append(kl_div)


def get_input(module, input, output):
    global input_images
    input_images.append(input[0].detach().cpu())


def save_images():
    for idx, image in enumerate(input_images):
        np.save(f'./saved_data/input_images/image_{idx}', image.numpy())


def fold_to_patches_uniform(feature_map, patch_division):
    B, C, H, W = feature_map.shape
    div_h = H // patch_division[0]
    div_w = W // patch_division[1]
    patches = feature_map.unfold(2, div_h, div_w).unfold(3, div_h, div_w)
    patches = patches.contiguous()
    # shape of patches at this point: (B, C, H//patch size, W//patch size, patch size, patch size)
    return patches


def calculate_stats(all_patches):
    patches_means = []
    patches_stds = []
    for patches in all_patches:
        patches_means.append(patches.mean(dim=[3, 4]))  # mean over the patch (squeeze only with B = 1)
        patches_stds.append(patches.std(dim=[3, 4]))
    return patches_means, patches_stds


def save_feature_maps():
    for layer_name, feature_map in feature_maps.items():
        np.save(f'./saved_data/feature_maps/{layer_name}_feature_map.npy', feature_map.cpu().detach().numpy())


def save_patch_stats():
    for stat, layer in patch_feature_maps.items():
        for layer_name, layer_maps in layer.items():
            print('layer size:', layer_maps.shape)
            np.save(f'./saved_data/patch_stats/{layer_name}_patch_{stat}.npy', layer_maps[0][0].cpu().detach().numpy())


def save_training_stats(module, name, last_conv):
    np.save(f'./saved_data/train_stats/{name}_mean.npy', module.running_mean.cpu().numpy())
    np.save(f'./saved_data/train_stats/{name}_var.npy', module.running_var.cpu().numpy())
    stats = {
        'Layer Name': name,
        'Last Convolution Layer': last_conv,
        'Running Mean file': f'./saved_data/train_stats/{name}_mean.npy',  # Associated file
        'Running Variance file': f'./saved_data/train_stats/{name}_mean.npy'  # Associated file
    }
    return stats


def kl_divergence(mu_p, sigma_p, mu_q, sigma_q):
    epsilon = 1e-8  # Small constant
    if sigma_q.any() < epsilon:
        print('training std very small->regularization')
    if sigma_p.any() < epsilon:
        print('patch std very small->regularization')
    sigma_q_regularized = np.maximum(sigma_q, epsilon)
    sigma_p_regularized = np.maximum(sigma_p, epsilon)

    sigma_p_sq = sigma_p_regularized ** 2
    sigma_q_sq = sigma_q_regularized ** 2

    kl_div = np.log(sigma_q_regularized / sigma_p_regularized) + (sigma_p_sq + (mu_p - mu_q) ** 2) / (2 * sigma_q_sq) - 0.5
    return kl_div


def compare_stats(training_stats, batchnorm_stats_df):
    global kl_divs
    for layer in batchnorm_stats_df['Last Convolution Layer']:
        #print(layer)
        bn_layer = batchnorm_stats_df[batchnorm_stats_df['Last Convolution Layer'] == layer]['Layer Name'].iloc[0]
        # Dynamically reshape based on the first dimension of training_stats['mean'][layer]
        first_dimension_size = training_stats['mean'][bn_layer].shape[0]
        reshaped_train_mean = training_stats['mean'][bn_layer].reshape((first_dimension_size, 1, 1))
        reshaped_train_std = np.sqrt(training_stats['var'][bn_layer]).reshape((first_dimension_size, 1, 1))

        kl_div = kl_divergence(patch_feature_maps['mean'][layer][0], patch_feature_maps['std'][layer][0],
                                      reshaped_train_mean, reshaped_train_std)

        if layer not in kl_divs:
            kl_divs[layer] = []
        kl_divs[layer].append(kl_div)


def save_kl_div():
    global kl_divs
    for layer in kl_divs:
        np.save(f'./saved_data/kl/kl_div_{layer}', kl_divs[layer])