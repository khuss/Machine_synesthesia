# Machine Synesthesia : using images for speech recognition 

The goal of this project is to build a voice recording digit classifier using the spoken digits dataset available [here](https://www.tensorflow.org/datasets/catalog/spoken_digit).

The classifier takes as an input any wave recording and will perform a speech/keyword detection algorithm outputting what digit is being spoken on the input recording.

The key part is using computer-vision techniques such as convolutional neural network to perform prediction, by leveraging the power and utility of mel spectrograms, as seen  in the figure below, we are able to teach the machine "to see the sound recording" (an example of synesthesia).
<p align="center">
<img src="https://github.com/khuss/Machine_synesthesia/blob/main/images/melspectrogram%20_0.png" width="400">
</p>

## Get started  

Create a virtual environment (It is preferable as not avoid any dependecy problems) and install the required packages as mentioned in the `requirements.txt` file.

`pip install -r requirements.txt`

Once all the required packages are installed open the `spoken_digits` notebook and follow the the instructions to either train the model from scratch or load and test the trained model.


## Prepare the data

To prepare the dataset, first using `tensorflow_datasets` download the dataset and  split as needed. 

Take a moment to explore the dataset after reading more about it using the `utils.py` functions or by yourself.

The function `preprocess.py` will be particularely useful to convert all the recordings to a uniformly sized spectrogram images that will served as input for the CNN model.

## Train the model

Using the Keras library interface, we then build a sequential model with multiple `CONV2D + MAXPOOling` blocks.

Feel free to customize the layers and hyperparameters to see if you can get a higher accuracy! 

When satisfied save yout model as done in the notebook. You will be able to load it again and use for prediction. You can also add a callback so that you can train it again.

## Test on your own recordings

Finally, you can test the model on your own recordings for fun.

As described in the last part of the notebook, upload your sample recordings into the directory and load them using `librosa.load('path_to_filename', 'sample_rate')` and specify the sample rate accordingly.

Then load your model, test and enjoy :) .


