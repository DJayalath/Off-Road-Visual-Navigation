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
# Predicts regional costs, computes trajectories, and commands the rover
# using the tflite model.
# Inputs:
# - model.tflite -> generated by tflite.py
#-------------------------------------------------------------------------------

from tflite_utils import NEW_STRIPS, margin_factor
import tflite_runtime.interpreter as tflite
from movement_servo import *
import numpy as np

# !: Keep consistent. Should be square root of N in the paper.
NUM_STRIPS = 8
NEW_STRIPS = 8

# Load tflite model
def load_tflite(fname='model.tflite'):

    interpreter = tflite.Interpreter(model_path=fname)
    interpreter.allocate_tensors()

    return interpreter

def tflite_predict(interpreter, img):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    batch = np.array([img])
    interpreter.set_tensor(input_details[0]['index'], batch)
    interpreter.invoke()

    indices = []
    state = {}
    inv = {}
    names = []
    for i in range(len(output_details)):
        index = output_details[i]['index']
        name = output_details[i]['name']
        state[index] = name
        inv[int(name.replace('StatefulPartitionedCall:', ''))] = index
        indices.append(output_details[i]['index'])
        names.append(int(name.replace('StatefulPartitionedCall:', '')))

    # Reorder outputs of model. See Appendix D in the paper.
    mapping = [54, 27, 28, 47, 10, 53, 14, 42, 30, 4, 56, 1, 21, 24, 15, 19, 22, 3, 39, 44, 29, 11, 8, 40, 52, 45, 20, 32, 23, 12, 25, 18, 57, 43, 55, 5, 31, 61, 46, 63, 59, 2, 51, 41, 0, 9, 36, 6, 33, 50, 62, 7, 58, 35, 16, 34, 37, 17, 48, 38, 13, 60, 49, 26]
    costs = []
    for i in range(NUM_STRIPS ** 2):
        output_data = interpreter.get_tensor(output_details[mapping[i]]['index'])
        costs.append(output_data[0][0])
        
    costs = np.array(costs)

    return costs

# Credit: Bruno Degomme, StackOverflow, 2021, [Online], Accessed on: 27 April 2022
# Available: https://stackoverflow.com/a/69141497
import cv2, threading
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()
    def _reader(self):
        while True:
            ret = self.cap.grab()
            if not ret:
                break
    def read(self):
        ret, frame = self.cap.retrieve()
        return frame

# Predicts a trajectory and asks if you want to follow it with the rover.
# If yes, rover will be instructed to follow.
def continuous_feed(interpreter):

    import time
    from timeit import default_timer as timer
    
    # Connect to vehicle
    setup()
    
    # Connect to webcam
    cap = VideoCapture(-1) 
    
    i = 0
    while True:

        # Read frame from webcam
        frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Allow time for webcam to balance lighting and capture a few frames
        if i < 3:
            i += 1
            time.sleep(1)
            continue
        
        i += 1
        
        # Predict costs and normalise
        costs = tflite_predict(interpreter, preprocess(frame).astype(np.float32))
        c_grid = norm_costs(costs)

        # Adjust for width
        c_grid = margin_factor(c_grid, 7)
        
        # Find top path
        start = timer()
        from path_finder import get_path
        paths, costs = get_path(c_grid.reshape(NEW_STRIPS, NEW_STRIPS), NEW_STRIPS)
        ids = np.argsort(costs)[:1].astype(int)
        paths = paths[ids]
        end = timer()
        print(f"\n----\nPath finding time = {end - start}\n----\n")
        
        # Show predicted path
        visualise(copy.deepcopy(frame), paths, c_grid)
        
        # Convert best path to heading
        headings = convert_path(paths[0])
        
        # If invalid, go to next frame.
        if None in headings:
            continue
        
        print("HEADING -->", headings)

        follow = input("follow? y/n: ")

        if follow == 'y':
        
            # Follow heading with vehicle 
            follow_headings(headings)
        
        plt.pause(0.001)

# Continuous predict --> follow --> predict cycle
# Keyboard Interrupt will stop the rover and then exit.
def free_travel(interpreter):

    try:
        import time
        from timeit import default_timer as timer
        
        # Connect to vehicle
        setup()
        
        cap = VideoCapture(-1)
        
        i = 0
        while True:

            frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Allow time for camera balancing
            if i < 3:
                i += 1
                time.sleep(1)
                continue
            
            i += 1
            
            # Predict costs, normalise, and visualise.
            costs = tflite_predict(interpreter, preprocess(frame).astype(np.float32))
            c_grid = norm_costs(costs)

            # Adjust for width
            c_grid = margin_factor(c_grid, 7)
            
            # Top 3 paths
            start = timer()
            from path_finder import get_path
            paths, costs = get_path(c_grid.reshape(NEW_STRIPS, NEW_STRIPS), NEW_STRIPS)
            ids = np.argsort(costs)[:1].astype(int)
            paths = paths[ids]
            end = timer()
            print(f"\n----\nPath finding time = {end - start}\n----\n")
            
            # Convert best path to heading
            headings = convert_path(paths[0])
            
            if None in headings:
                continue
            
            print("HEADING -->", headings)

            follow_headings(headings)

    except KeyboardInterrupt:
        # Stop the vehicle!
        stop_servos()
        raise

if __name__ == "__main__":
    

    from tflite_utils import visualise, preprocess, convert_path, norm_costs

    import cv2
    import copy

    import matplotlib.pyplot as plt
    interpreter = load_tflite()
    continuous_feed(interpreter)
