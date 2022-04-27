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
# Adapted depth map denoising method from 
# https://dev.intelrealsense.com/docs/depth-post-processing
#-------------------------------------------------------------------------------

import numpy as np
import scipy as sp

# Non-zero mean sub-sampling with a 4x4 window
def subsample(depth):
    h, w = depth.shape
    buf = np.empty(depth.shape)
    s = 4
    for y in range(0, h, s):
        for x in range(0, w, s):
            buf[y : y + 4, x : x + 4] = np.nanmean(depth[y : y + 4, x : x + 4])
    return buf

# Gaussian filtering
# Credit: David, StackOverflow, 2013, [Online], Accessed on: 27 April 2022
# Available: https://stackoverflow.com/a/36307291
def gaussian_filter(buf):
    sigma=2.0                  # standard deviation for Gaussian kernel
    truncate=4.0               # truncate filter at this many sigmas

    V=buf.copy()
    V[np.isnan(buf)]=0
    VV=sp.ndimage.gaussian_filter(V,sigma=sigma,truncate=truncate)

    W=0*buf.copy()+1
    W[np.isnan(buf)]=0
    WW=sp.ndimage.gaussian_filter(W,sigma=sigma,truncate=truncate)

    Z=VV/WW
    buf = Z

    return buf

# Fill holes in the depth map (i.e. zero or NaN)
def patch_holes(buf):
    h, w = buf.shape
    for y in range(h):
        for x in range(w):
            r = 1
            while np.isnan(buf[y, x]):

                if r > 10:

                    # Look left and up, cascading failure cases.

                    # Look left
                    for i in range(x):
                        if not np.isnan(buf[y, x - i]):
                            buf[y, x] = buf[y, x - i]
                    # Look up
                    if np.isnan(buf[y, x]):
                        for i in range(y):
                            if not np.isnan(buf[y - i, x]):
                                buf[y, x] = buf[y - i, x]
                    # Median of row
                    if np.isnan(buf[y, x]):
                        buf[y, x] = np.nanmedian(buf[y, :])
                    # Median of column
                    if np.isnan(buf[y, x]):
                        buf[y, x] = np.nanmedian(buf[:, x])
                    # Mean of image
                    if np.isnan(buf[y, x]):
                        buf[y, x] = np.nanmean(buf)

                else:

                    # Fill depth hole by scanning for non-NaNs in increasing
                    # window around the hole.
                    med = np.nanmedian(buf[y - r : y + 1, x - r : x + 1])
                    buf[y, x] = med
                    r += 1
    
    return buf

# Input [0, 255] depth map
# Returns denoised [0, 255] depth map
def denoise(depth):

    h, w = depth.shape

    # Convert to FP and normalise
    img = depth
    img = img.astype(np.float32) / 255.

    # Convert all zeros to nans
    for y in range(h):
        for x in range(w):
            if img[y, x] == 0:
                img[y, x] = np.nan

    buf = subsample(img)
    buf = gaussian_filter(buf)
    buf = patch_holes(buf)
    
    return (buf * 255.0).astype(np.uint8)