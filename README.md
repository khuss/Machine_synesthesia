# Machine Synesthesia

The goal of this project is to build a voice recording digit classifier using the spoken digits dataset available [here](https://www.tensorflow.org/datasets/catalog/spoken_digit).

The classifier takes as an input any wave recording and will perform a speech/keyword detection algorithm outputting what digit is being spoken on the input recording.

The key part is using computer-vision techniques such as convolutional neural network to perform prediction, by leveraging the power and utility of mel spectrograms, as seen  in the figure below, we are able to teach the machine "to see the sound recording" (an example of synesthesia).


## Get started  

Create a virtual environment (It is preferable as not avoid any dependecy problems) and install the required packages as mentioned in the `requirements.txt` file.

`pip install -r requirements.txt`



