# Project-Kodo
The goal of Project Kodo was originally to be an autonomous driving agent for GTA V. As the idea progressed however, it became more of a platform for easy training and usage of machine learning agents in games in general. It's constructed using a modular architecture that aims to allow easy addition of models, input devices and action broadcasting methods.


## Examples
I've used the platform to train an autonomous driving agent for GTA V. Below are some results using an implementation of [NVIDIA's autopilot model](https://arxiv.org/pdf/1604.07316.pdf). The agent drives solely based on image data without any other inputs from the game.

The model was trained on about an hour of training data, which was collected by driving around the game world. During pre-processing, frames with little turning are dropped, as per the original paper. To do this, a [logistic function](https://www.wolframalpha.com/input/?i=(1+%2F+(+1+%2B+exp(-15*(x+-+0.4)))),+plot+from+0+to+1) is used to weigh the magnitude of the left stick's X-axis in each frame, and frames where the magnitude is larger, are more likely to not get dropped out. This turned out to be VERY important. The model seen in the video below was trained using a batch size of 32 for 300 epochs. The a training dataset consisted of about 2500 frames, picked from a total of 30000 images. Though getting the model right took quite a while, training took only 15 minutes on an NVIDIA P5000.

As can be seen, the agent does make some mistakes. Still, it seems to be able to recover surprisingly well, considering that the training data consists of fairly "clean" driving. Future work includes adding traffic and pedestrians to the mix, and recording more training data to get a more accurate model.
 
 #### Video:
[![Video of autonomous driving](https://img.youtube.com/vi/hISzqO2uPwo/0.jpg)](https://www.youtube.com/watch?v=hISzqO2uPwo)

## Usage
In its current state, the GUI is meant to simply aid the training process by providing a quick interface. It's quite ugly and it WILL break if invalid data is provided. That being said, I've tried to remove obvious sources of user error, and a user with some knowledge of the underlying process should be able to use the program without any problems.


### Installation
- 64-bit version of Python is required!
- Tested on `python 3.6.4` on Windows 10, MacOS Sierra and Ubuntu 14.04
- Install [Microsoft Visual C++](http://landinghub.visualstudio.com/visual-cpp-build-tools)
- Run `pip install -r requirements` to install the required python packages
- If you have a system with a [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus), follow the instructions [here](https://www.tensorflow.org/install/) to install tensorflow-gpu
- Xbox 360 wireless controller drivers need to be installed for recording. The specific driver depends on your platform.
- [vJoy](http://vjoystick.sourceforge.net) required for prediction (Only works on Windows)
- [x360ce](http://www.x360ce.com) is also required for prediction. (Only works on Windows as well)


### Running
Run `python run.py` in the project's root folder


### Recording

![recording](/screenshots/recording.png?raw=true)

The recording screen shows a live feed of the captured area of the screen, resized to the actual recording's output resolution. Next to it, controller outputs are shown, and below it, a menu that can be used to alter recording settings is provided. Prediction controls are located in the same tab.

The following recording parameters can be set:
- **Capture screen size**: The size of the capture area. The upper-left corner of the area is situated 40 pixels below the upper-left corner of the screen, to crop out the frames of windows.
- **Recording resolution**: Resolution of the output images
- **Input device**: The device, the output of which is used as the target of the model
- **Recording save directory**: The folder to which the recorded data is saved. **NOTE:** The data has to be located in `Project-Kodo/data` to be used in later steps.



### Processing
![processing](/screenshots/processing.png?raw=true)
The processing tab provides an easy way to generate training data for a specific model. A screen, rendering the processed data, is shown on the left during processing, though it can be turned off to improve performance. On the right, a list of checkboxes next to each controller channel can be used to keep only certain inputs in the training data.



### Training
![training](/screenshots/training.png?raw=true)
In the training tab, the user can set basic hyperparameters for training the model, as well as a name for the weights. Multiple processed datasets can be chosen to be trained on. Starting the training opens a Tensorboard console that can be used for following the training process. **NOTE:** In the beginning the page will probably not work. Make sure to refresh the page after the first training epoch has completed to see the actual console.



### Predicting
**Prediction currently only works on Windows!** This is due to the limitations of setting up virtual controllers on other systems. Linux support seems to be a viable feature addition in the future, but unfortunately MacOS isn't happening any time soon.

Prediction can be done in the same tab as recording. To start predicting run x360ce, select the model and the associated set of weights to be used. The platform will then capture the image from the defined area and use the model to predict the outputs. Finally it uses PyvJoy and x360ce to emit the controls to the game.



## Adding models
The model API is still under work. Currently each model needs to subclass `models.template.KodoModel` and have the following methods:

- `process(data_folder, input_channels_mask, img_update_callback=None)`, which processes the data to conform with the model's required input and saves the result to models/[model_name]/data/[data_folder_name]
- `create_model()`, which creates the actual model and sets it to `self.model`
- `train(batch_size, epochs, weights_name)`, which trains the model and saves the resulting set of weights and associated info to `models/[model_name]/weights/[weights_name]`
- `get_actions(img)`, which returns the model's prediction given an input image

If those requirements are met, the model is ready to be trained and used with the platform!



## Future Work
- Make it easier to add new controllers/keyboards
- Add keyboard input
- Add more complex models
- Add ETA or some other indication of how much of the processing is done
- Make model API clearer
- Make GUI nicer
- Make code prettier, and add comments and documentation to make contributing easier
- Show info about selected dataset on training screen

## Sources of inspiration and help
- [Tensorkart](https://github.com/kevinhughes27/TensorKart)
- [pygta5](https://github.com/Sentdex/pygta5)




