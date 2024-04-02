feature_maps = []

def get_feature_maps(module, input, output):
    global feature_maps
    feature_maps.append(output.detach())

def fold_to_patches(patch_size):
    all_patches = []
    for feature_map in feature_maps:
        B, C, H, W = feature_map.shape
        patches = feature_map.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous()
        # shape of patches at this point: (B, C, H//patch size, W//patch size, patch size, patch size)
        all_patches.append(patches)
    return all_patches

def calculate_stats(all_patches):
    patches_means = []
    patches_stds = []
    for patches in all_patches:
        patches_means.append(patches.mean(dim=[4, 5]).squeeze(0))  # mean over the patch (squeeze only with B = 1)
        patches_stds.append(patches.std(dim=[4, 5]).squeeze(0))
    return patches_means, patches_stds
