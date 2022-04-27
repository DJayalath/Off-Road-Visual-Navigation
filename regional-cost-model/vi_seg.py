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
# Computes vegetation costs from an RGB image.
# Inputs:
# - ndvi_predictor.joblib -> generated by ndvi-model/model.py
# - ndvi_scaler.joblib -> generated by ndvi-model/model.py
#-------------------------------------------------------------------------------

import numpy as np
import cv2

# Load models
from sklearn.linear_model import SGDRegressor
from joblib import load

ndvi_reg = load('ndvi_predictor.joblib')
ndvi_scaler = load('ndvi_scaler.joblib')

# Input rgb image
# Output is vegetation cost response map
def vi_cost(rgb_image):

    # --- Estimate NDVI ---

    # Scale image
    h, w = rgb_image.shape[:2]
    scaled_x = ndvi_scaler.transform(rgb_image.reshape(h * w, -1))

    # Predict
    y = ndvi_reg.predict(scaled_x)

    # Post-processing and scaling to [0, 1] range
    y = np.clip(y, -1, 1)
    y = (y.astype(np.float32) + 1) / 2

    # --- Threshold NDVI response to segment sky ---

    # Otsu's thresholding after Gaussian filtering
    vi = (y * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(vi.reshape(480, 640),(5,5),0)
    th,vi_classified = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th *= 0.75
    th,vi_classified = cv2.threshold(blur,th,255,cv2.THRESH_BINARY)
    vi_classified = vi_classified.flatten()

    # Assume lower half is ZERO, upper half is 255

    # --- Compute VARI response ---

    h, w = rgb_image.shape[:2]
    rgb_image = rgb_image.astype(np.float32)
    r = rgb_image.reshape(h * w, -1)[:, 0]
    g = rgb_image.reshape(h * w, -1)[:, 1]
    b = rgb_image.reshape(h * w, -1)[:, 2]
    y = np.empty(len(r))
    for i in range(len(r)):
        if g[i] + r[i] - b[i] == 0:
            y[i] = 0
        else:
            y[i] = (g[i] - r[i]) / (g[i] + r[i] - b[i])
    y = np.clip(y, 0, 1)
    vari = y

    # --- Construct cost map ---

    cost = np.zeros(len(vi_classified))
    for i in range(len(vi_classified)):
        if vi_classified[i] == 255:
            cost[i] = 1 - vari[i]

    # BOOST signal contrast by squaring
    cost = np.power(cost, 2)

    return cost.reshape(h, w)