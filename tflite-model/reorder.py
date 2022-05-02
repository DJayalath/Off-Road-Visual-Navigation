################################################################################
#
# This work is all the original work of the author and is distributed under the
# GNU General Public License v3.0.
#
# Created By : Dulhan Jayalath
# Date : 2022/05/02
# Project : Real-time Neural Visual Navigation for Autonomous Off-Road Robots
#
################################################################################

# ------------------------------------------------------------------------------
# A utility that compares the order of two lists and produces a mapping from
# the first list to the second. Use this to resolve the model output order
# problem described in Appendix D of the paper.
#-------------------------------------------------------------------------------

import numpy as np

# FIXME: Insert the patch predictions for a reference image predicted by
# a full TensorFlow model and its TFLite equivalent here. The TFLite model must
# be run using TFLite and not TensorFlow's TFLite interpreter.
tensorflow = np.array([])
tflite = np.array([])

# Prints the mapping of the first list to the second. Copy this to the variable
# named 'mapping' in tflite_standalone.py to resolve output order matching.
print("[", end='')
for i in range(len(tensorflow)):
    idx = (np.abs(tflite - tensorflow[i])).argmin()
    print(f"{idx}, ", end='')
print("]")