################################################################################
#
# This work is all the original work of the author and is distributed under the
# GNU General Public License v3.0.
#
# Created By : Dulhan Jayalath
# Date : 2022/04/27
# Project : Real-time Neural Visual Navigation for Autonomous Off-Road Robots
#
################################################################################

# ------------------------------------------------------------------------------
# Labels the low-viewpoint forest depth dataset with regional cost labels
# Inputs:
# - Low-viewpoint forest depth dataset -> https://zenodo.org/record/3945526
# Outputs:
# - labels_train.json / labels_test.json -> labels for images
# Requirements:
# - Extract all the RGB files in the dataset into one folder called 'rgb'
#   and all the depth maps into another folder at the same level called 'depth'
# - (optional) Split the dataset into a train and test set with structure
#   train/rgb, train/depth, test/rgb, test/depth
#-------------------------------------------------------------------------------

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

# FIXME: Change this to directory of Low-viewpoint forest depth dataset rgb files.
# May need to change 'train' to 'test' when labelling test data.
RGB_FOREST = 'train/rgb'
DEPTH_FOREST = 'train/depth'

# Ensure this is the square root of N in the paper.
ROOT_N = 8

# Computes a level of obstruction for a given input (most likely a vertical pixel rectangle).
def cost(depth, vi):

    h, w = depth.shape[:2]
    depth = depth.reshape(h * w).astype(np.float32)

    vi = vi.reshape(h * w)

    # Compute depth cost, throwing away invalid estimates.
    depth_cost = np.array(list(map(lambda x : ((255.0 - x) / 255.0), depth)), dtype=np.float32)

    # Compute f_c in paper for every pixel.
    cvi = np.zeros(h * w)
    for i in range(len(depth_cost)):
        if depth_cost[i] == 0:
            cvi[i] = vi[i]
        if depth_cost[i] == -1:
            depth_cost[i] = 0

    cost_map = vi * depth_cost + 0.1 * cvi

    return cost_map.reshape(h, w)

# Aggregate costs for every patch
def cost_grid(depth, vi, g_mask):

    h, w = depth.shape
    stride_y = h // ROOT_N
    stride_x = w // ROOT_N

    cost_map = cost(depth, vi)

    if g_mask is not None:
        cost_map *= g_mask

    c_grid = np.empty((ROOT_N, ROOT_N), dtype=np.float32)
    for i, y in enumerate(range(0, h, stride_y)):
        for j, x in enumerate(range(0, w, stride_x)):
            c_grid[i, j] = np.sum(cost_map[y : y + stride_y, x : x + stride_x])

    return c_grid

# Compute regional costs for an image
# Returns:
# - d -> denoised depth map
# - vi -> vegetation costs
# - c_grid -> regional costs
def predict_image(image, depth):

    from depth_denoise import denoise
    from vi_seg import vi_cost
    from ground_seg import segment_ground
    import copy

    g_mask = segment_ground(copy.deepcopy(depth))

    d = denoise(depth)

    vi = vi_cost(image)

    c_grid = cost_grid(d, vi, g_mask)

    if g_mask is not None:
        vi = vi * g_mask.astype(np.float32)

    return d, vi, c_grid

# Label an image with its regional costs
def label(im_path):

    # Read and predict
    image = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(im_path.replace('/rgb', '/depth'), cv2.IMREAD_GRAYSCALE)
    _, _, c_grid = predict_image(image, depth)
    c_grid = c_grid.flatten()

    # For each image, we create a dictionary with cost for each patch
    record = {}
    for i in range(ROOT_N ** 2):
        record[f'cost_{i}'] = float(c_grid[i])
    
    im_dict = {
        os.path.basename(im_path) : record
    }

    return im_dict

# Label the dataset with regional costs
def label_all():

    paths = glob.glob(f'{RGB_FOREST}/*.png')

    labels = {}

    import warnings
    warnings.filterwarnings('ignore')

    import concurrent.futures
    from tqdm import tqdm
    
    with concurrent.futures.ProcessPoolExecutor(16) as executor:

        print("Making predictions...")
        results = list(tqdm(executor.map(label, paths), total=len(paths)))

        print("Tabulating predictions...")
        for result in results:
            labels.update(result)
        
        # Scale costs to [0, 1] range for stability with MAE
        print("Finding max and min costs...")
        vmin, vmax = 99e9, 0
        for _, val in labels.items():
            for i in range(ROOT_N):
                v = val[f'cost_{i}']
                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v
        print("(Min, Max): ", vmin, vmax)
        
        print("Writing predictions...")
        import json

        # FIXME: May need to change 'train' to 'test'
        with open(f'./labels_train.json', 'w') as f:
            json.dump(labels, f)
        
if __name__ == "__main__":
    label_all()