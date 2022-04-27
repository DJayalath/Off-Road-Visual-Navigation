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
# A Keras Sequence generator to process RGB images and regional cost labels
# for use with the Keras API.
# See https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
# and usage in tf.keras.Model.fit()
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
#-------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import random
from PIL import Image
import os

class RegressionGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_folder, label_file_path, preprocessor, batch_size):

        self.batch_size = batch_size
        self.preprocessor = preprocessor
        
        import json
        self.labels = json.load(open(f'{label_file_path}'))

        import glob
        self.x_paths = glob.glob(f'{x_folder}/*.png')

        random.shuffle(self.x_paths)

    def __len__(self):
        return int(np.ceil(len(self.x_paths) / float(self.batch_size)))

    def __getitem__(self, idx):

        paths = self.x_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([self.preprocessor(np.array(Image.open(path))) for path in paths])
        

        y = {}
        fname = os.path.basename(paths[0])
        tmp_y = self.labels[fname]
        for key, _ in tmp_y.items():
            vals = []
            for path in paths:
                fname = os.path.basename(path)
                vals.append(self.labels[fname][key])
            y[key] = np.array(vals)

        return batch_x, y