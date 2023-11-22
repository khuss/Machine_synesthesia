import matplotlib.pyplot as plt
from IPython.display import Audio, display
import librosa
import numpy as np

def render_history(history):
    plt.plot(history["loss"], label="loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Our losses")
    plt.show()
    plt.close()

    plt.plot(history["accuracy"], label="accuracy")
    plt.plot(history["val_accuracy"], label="val_accuracy")
    plt.legend()
    plt.title("Our accuracies")
    plt.show()
    plt.close()


def compare_histories(history_list):
    for training_name, history in history_list.items():
      plt.plot(history["val_accuracy"], label=training_name)
      plt.legend()
      plt.title("Comparision of val_accuracy")
      plt.show()
      plt.close()


def display_wave(sample):
    audio = sample["audio"].numpy().astype("float32")
    label = sample["label"].numpy()
    plt.plot(audio)
    plt.title(f"Label is {label}")
    plt.show()
    plt.close()
    
def play_audio(sample, sr = 8000):
    audio = sample["audio"].numpy().astype("float32")
    display(Audio(audio, rate=sr))
    
def display_spectrogram(sample):
    mel = librosa.feature.melspectrogram(
        y=sample["audio"].numpy().astype("float32"),n_mels=64,hop_length=64,sr=8000,fmax=2000
    )
    mel /= np.max(mel)
    plt.imshow(mel[::-1,:], cmap="inferno")