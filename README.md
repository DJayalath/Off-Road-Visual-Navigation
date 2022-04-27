# Real-Time Neural Visual Navigation for Autonomous Off-Road Robots

This work provides a method for real-time and low-cost visual navigation for ground robots in unstructured environments. The algorithm uses a convolutional network to propose a set of trajectories and costs for efficient and collision-free movement from a single webcam. The models in this project are trained to specifically target forests. For more details, see the paper.

## Quick Start
1. Generate an NDVI prediction model from `ndvi-model/model.py`
2. Label datasets using `regional-cost-model/label.py`
3. Train the model with `regional-cost-model/model.py`
4. Convert the model to TFLite with `tflite-model/tflite.py`
5. Run the TFLite model online with `tflite-model/tflite_standalone.py`

Notes: Prior to step 5, it may be necessary to resolve the order of the model's output heads. See Appendix D of the paper. For further information on any of the steps, see the paper and the included comments in the code.