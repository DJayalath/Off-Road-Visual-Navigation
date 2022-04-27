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
# Ground segmentation method implemented based on Kırcalı et al.
# See [KT14] and Appendix C.
#-------------------------------------------------------------------------------

from numpy import argmax
from scipy.optimize import curve_fit
from scipy import ndimage
import numpy as np

# Depth envelope detection
def envelope(depth):
    h, w = depth.shape

    envelope = []

    # Starting from bottom row, pick furthest (maximum) value
    # (throws away invalid pixels (0) by default)
    for y in range(h):
        envelope.append(depth[h - y - 1, :].max())

    # Truncate envelope following peak
    envelope = envelope[:argmax(envelope)]

    # Median filtering to remove outliers before fitting
    from scipy import ndimage
    envelope = ndimage.median_filter(envelope, size=9)

    return envelope

# Double exponential function
def fit_func(x, a, b, c, d):
    return a * np.exp(b * x) + c * np.exp(d * x)

# Fit envelope to double exponential function, returning parameters and error
def fit(envelope):
    x = np.array(list(range(len(envelope))), dtype=np.float64)
    params = curve_fit(fit_func, x, envelope, maxfev=1000000, p0=[0.01, 0.01, 0.01, 0.001])

    a,b,c,d = params[0]

    error = 0
    for i in range(len(envelope)):
        error += (envelope[i] - fit_func(i, a, b, c, d)) ** 2

    return ((a, b, c, d), error)

# Remove ground based on closeness to fitted curve with threshold
def chop_ground(depth, params, thresh = 0.1):
    h, w = depth.shape

    for y in range(h):
        for x in range(w):

            # Ignoring invalid depth values (0) by default
            ref = fit_func(h - y - 1, params[0], params[1], params[2], params[3])
            act = depth[y, x]

            if abs(act - ref) < thresh:
                depth[y, x] = -1

    return depth

# Compensate for roll in the image by finding the rotation
# with the minimum dissimilarity in rank depth values.
def roll_comp(depth):

    h, w = depth.shape

    min_dis = 99e9
    best_rot = -1
    for theta in range(-15, 16):

        rotated = ndimage.rotate(depth, theta, reshape=False, order=0)

        # Compute similarity in each row
        dissimilarity = 0
        for y in range(h):
            dissimilarity += np.std(rotated[y, :])

        if dissimilarity < min_dis:
            min_dis = dissimilarity
            best_rot = theta
    
    return best_rot

# Expects depth image [0, 255]
# Returns mask with 0s for ground and 1s o/w
# Returns None if ground segmentation fails
def segment_ground(depth):

    try:
        h, w = depth.shape
        d = depth.astype(np.float32) / 255.

        # Roll correction
        theta = roll_comp(depth)
        if theta != 0:
            depth = ndimage.rotate(depth, theta, reshape=False, order=0)
        

        params, error = fit(envelope(d))

        if error > 0.3:
            raise Exception("Error > 0.3")

        chop_depth = chop_ground(d, params)

        # Correct roll compensation back to original orientation
        chop_depth = ndimage.rotate(chop_depth, -theta, reshape=False, order=0)

        # Generate mask
        chop_depth[chop_depth != -1] = 1
        chop_depth[chop_depth == -1] = 0

        return chop_depth.astype(np.float32)

    except Exception as _:
        return None