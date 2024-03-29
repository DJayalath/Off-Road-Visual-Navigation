# Real-Time Neural Visual Navigation for Autonomous Off-Road Robots

This work provides a method for real-time visual navigation for ground robots in unstructured environments. The algorithm uses a convolutional network to propose a set of trajectories and costs for efficient and collision-free movement from a single webcam. The models in this project are trained to specifically target forests. For more details, see the [paper](https://dulhanjayalath.com/report-compressed.pdf).

<p align="center">
  <img src="images/prediction_example.png" />
  <br>
  Figure: Top three trajectories (green, yellow, red) predicted from a forest scene.
</p>

## Quick Start
Setup: Clone the repository and follow the instructions in the headers of each of the files listed below.

1. Generate an NDVI prediction model from `ndvi-model/model.py`
2. Label datasets using `regional-cost-model/label.py`
3. Train the model with `regional-cost-model/model.py`
4. Convert the model to TFLite with `tflite-model/tflite.py`
5. Run the TFLite model online with `tflite-model/tflite_standalone.py`

Notes: Prior to step 5, it may be necessary to resolve the order of the model's output heads. See Appendix D of the paper and `tflite-model/reorder.py`. For further information on any of the steps, see the paper and included comments in the code.

## Demo
[![Demo video](images/cropped_movie_link_play.jpg)](https://www.youtube.com/watch?v=ktvmSO5Y_PE)

## Requirements
### Training
- TensorFlow 2.x
- OpenCV 4.x
- scikit-learn 1.x
### Inference
- TFLite 2.x
- SciPy 1.x
### Control (optional via Pixhawk Autopilot)
- DroneKit 2.x
- pymavlink 2.x 

Other miscellaneous libraries (e.g. matplotlib) will be needed to run the code but are not required.

## Test Configuration from Paper
- Hardware: Raspberry Pi 4B (TBC GiB)
- Traversal Cost Labelling Parameters: (γ, α, β) = (0, 1, 0.1)
- Model Parameters:
  - Optimizer: Adam
  - Batch Size: 32
  - Loss: MAE
  - Epochs: 100
  - Train-Test Split: 80-20
  - Number of Instances: 9700

## License
This work is all the original work of the author and is distributed under the GNU General Public License v3.0. For more details, see the [LICENSE](LICENSE) file.
