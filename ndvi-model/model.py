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
# Trains a model with SGD to predict NDVI values from RGB pixels
# Inputs:
# - Freiburg forest dataset -> http://deepscene.cs.uni-freiburg.de/
# Outputs:
# - ndvi_predictor.joblib -> Trained model for NDVI prediction
# - ndvi_scaler.joblib -> Trained input scaler
#-------------------------------------------------------------------------------

# FIXME: Change this to point to Freiburg dataset rgb files
TRAIN_FILES = 'train/rgb/*.jpg'
TEST_FILES = 'test/rgb/*.jpg'

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from joblib import dump
import numpy as np
import random
import glob
import cv2

# Generate two lists of paths from a list of paths to RGB images in the Freiburg dataset.
# x_files contains a list of paths to RGB images.
# y_files will contain a list of paths to associated NDVI maps.
def gen_paths(x_files, limit=1000):
    random.shuffle(x_files)
    x_files = x_files[:limit]
    y_files = list(map(lambda x : x.replace('_Clipped.jpg', '_NDVI_Float.tif').replace('rgb', 'ndvi_float'), x_files))
    return x_files, y_files

# Load arrays of RGB pixels and associated NDVI values from lists of paths to 
# RGB images and paths to associated NDVI maps.
def load_data(x_files, y_files):
    
    X = []
    Y = []

    # Extract RGB pixel values
    for f in x_files:
        img = cv2.imread(f, cv2.IMREAD_ANYCOLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        for rgb in img.reshape(h * w, -1):
            X.append(rgb)

    # Extract NDVI values
    for f in y_files:
        img = cv2.imread(f, flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
        for px in img.flatten():
            Y.append(px)

    X, Y = np.array(X), np.array(Y)

    return unison_shuffled_copies(X, Y)

# Credit: mtrw, StackOverflow, 2011, [Online], Accessed on: 27 April 2022
# Available: https://stackoverflow.com/a/4602224
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

x_files, y_files = gen_paths(glob.glob(TRAIN_FILES), limit=50)
x_tfiles, y_tfiles = gen_paths(glob.glob(TEST_FILES), limit=10)

X_train, y_train = load_data(x_files, y_files)
X_test, y_test = load_data(x_tfiles, y_tfiles)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 5-fold cross-validation on training data
print("\n---- 5-Fold Cross Validation ----")
reg = SGDRegressor(max_iter=1000, tol=1e-3)
scores = cross_val_score(reg, X_train, y_train)
print(f"\n{scores.mean()} R^2 with a standard deviation of {scores.std()}\n")
print("----\n")

# Create new model fitted to all training data
print("\n---- Fitting to train data ----")
reg = SGDRegressor(max_iter=1000, tol=1e-3)
reg.fit(X_train, y_train)

# Compute R^2 scores on test and train set
print(f"\nR^2 Train: {reg.score(X_train, y_train)}")
print(f"R^2 Test: {reg.score(X_test, y_test)}\n")
print("----\n")

# Save model and scaler to disk
dump(reg, 'ndvi_predictor.joblib')
dump(scaler, 'ndvi_scaler.joblib')