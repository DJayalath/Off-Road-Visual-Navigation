################################################################################
#
# (c) Copyright University of Southampton, 2022
#
# Copyright in this software belongs to the University of Southampton,
# Highfield, University Road, Southampton, SO17 1BJ, United Kingdom
#
# Created By : Dulhan Jayalath
# Date : 2022/04/27
# Project : Real-time Neural Visual Navigation for Autonomous Off-Road Robots
#
################################################################################

# ------------------------------------------------------------------------------
# Various utilities for prediction and movement
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# Keep the same as square root of N in the paper.
NUM_STRIPS = 8
NEW_STRIPS = 8

N_WIDTH, N_HEIGHT = 640, 480

# Margin should be in range 1 - 8 given 8x8 cost grid
def margin_factor(c_grid, margin):
    c_grid = c_grid.reshape(NEW_STRIPS, NEW_STRIPS)

    # Each cell should collect neighbour costs in Gaussian fashion.
    # Make Gaussian square rather than straight
    c_grid = cv2.GaussianBlur(c_grid,(margin,margin),0)

    return c_grid.flatten()

# Preprocessing for model
def preprocess(x):
    x = cv2.resize(x, (224, 224))
    return x

# Normalise costs
def norm_costs(costs):
    c_grid = costs
    X_std = (c_grid - c_grid.min()) / (c_grid.max() - c_grid.min())
    c_grid = (X_std * 255.0).astype(np.uint8)
    return c_grid

# Visualises a trajectory prediction or set of predictions
# Inputs:
# - image -> RGB picture
# - paths -> suggested paths from path_finder.py
# - c_grid -> regional costs
def visualise(image, paths, c_grid):

    image=cv2.resize(image, (N_WIDTH, N_HEIGHT))

    h, w = image.shape[:2]
    x_stride = w // NEW_STRIPS
    y_stride = h // NEW_STRIPS

    cost_map = np.zeros(image.shape, dtype=np.uint8)

    i = 0
    for y in range(0, h, y_stride):
        for x in range(0, w, x_stride):
            cost_map[y : y + y_stride, x : x + x_stride, 0] = c_grid[i]
            i += 1

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    result = cv2.addWeighted(gray, 1.0, cost_map, 1.0, 0)

    colors = ['lime', 'yellow', 'red', 'blue', 'black', 'orange', 'white']
    z = 0
    plt.imshow(image, extent=[0, w, 0, h])
    for p in paths:
        # print("Path: ", p)
        # ax[1].imshow(image, extent=[0, w, 0, h])
        xs, ys = [], []
        for v in p:
            # Start node
            if v == (NEW_STRIPS * NEW_STRIPS):
                # print(w, w / 2, "START POS")
                xs.append(w / 2)
                ys.append(0)
            else:
                xs.append((w // NEW_STRIPS) * 0.5 + (v % NEW_STRIPS) * (w // NEW_STRIPS))
                ys.append(h - (v // NEW_STRIPS) * (h // NEW_STRIPS) - (h // NEW_STRIPS) * 0.5)  # - (h // NEW_STRIPS) * 0.5)

        # Obligatory starting pos behind camera
        xs.append(w / 2)
        ys.append(- h // NEW_STRIPS)
        
        for i in range(0, len(xs) - 1):
            plt.plot(xs[i:i+2], ys[i:i+2], '-', color=colors[z], linewidth=4.0)
        z += 1

    # Disable x and y ticks
    plt.xticks([])
    plt.yticks([])

    plt.axis((0, 640, 0, 480))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.tight_layout()
    plt.savefig(f'results/{timestr}.png', dpi=600, bbox_inches='tight')
    plt.show()

# Converts a path from indexes into headings e.g. N NW etc.
# Note that paths appear in reverse order (i.e. low indexes to high)
# 0 is top left of grid
def convert_path(idx_path):
    
    headings = []
    
    # Reverse path
    path = idx_path[::-1]

    # Starting position
    if path[0] == 59:
        headings.append('L_START')
    elif path[0] == 60:
        headings.append('R_START')
    
    # Read from bottom to top
    for i in range(len(path) - 1):
        
        head = None
        
        p1 = path[i]
        p2 = path[i + 1]
        
        if p2 == p1 - NEW_STRIPS:
            head = 'N'
        elif p2 == p1 - NEW_STRIPS + 1:
            head = 'NE'
        elif p2 == p1 - NEW_STRIPS - 1:
            head = 'NW'
        elif p2 == p1 + 1:
            head = 'W'
        elif p2 == p1 - 1:
            head = 'E'
            
        headings.append(head)
    
    return headings