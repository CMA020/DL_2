import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import os
import gc
from pydub import AudioSegment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
import csv
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
def generate_spectrogram(audio_path, output_dir):
    # Load the audio file
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_path)
    ms = librosa.feature.melspectrogram(y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    # Save the spectrogram as an image
    filename = os.path.splitext(os.path.basename(audio_path))[0]

    input_extension = os.path.splitext(os.path.basename(audio_path))[1]

    output_filename = f'{filename}{input_extension}.png'
    fig.savefig(os.path.join(output_dir, output_filename))

    gc.collect()

def convert_audio(input_file, output_file):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Export the audio as a WAV file
    audio.export(output_file, format="wav")


def create_model():


    model = Sequential([
        Conv2D(filters=128, kernel_size=(5, 5), padding='valid', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(filters=64, kernel_size=(5, 5), padding='valid', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),


        Conv2D(filters=32, kernel_size=(3, 3), padding='valid', kernel_regularizer=l2(0.00005)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Flatten(),

        Dense(units=512, activation='relu'),
        Dropout(0.5),
        Dense(units=13, activation='softmax')
    ])

    return model
IMG_WIDTH = 150
IMG_HEIGHT = 150
BATCH_SIZE = 32
dict={'0': 12, '1': 3 , '2':11,'3':13 , '4':4 , '5':5 , '6':1 , '7':7 , '8':10 , '9':2 , '10':8 , '11':6 , '12':9  }
if __name__ == '__main__':
    cnn_model = create_model()
    cnn_model.load_weights(os.path.expanduser("/content/model_weights2.h5"))
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    parent_dir = os.path.expanduser('/content/drive/MyDrive/Project/spectrograms/val')  ##### add test directory  path
    os.makedirs("~/DL_1/Test2")
    output_dir = os.path.expanduser("~/DL_1/Test2")

    # Iterate over directories inside the parent directory
    for audio_file in os.listdir(parent_dir):

        if audio_file.endswith('.wav') or audio_file.endswith('.mp3') or audio_file.endswith(
                '.aiff') or audio_file.endswith('.flac') or audio_file.endswith('.ogg') or audio_file.endswith(
            '.m4a'):
            audio_path = os.path.join(parent_dir, audio_file)
            generate_spectrogram(audio_path, output_dir)


    with open(os.path.expanduser('/content/drive/MyDrive/1.csv'), mode='w', newline='') as file:    ###W add csv file path
        writer = csv.writer(file)

        # Write header row
        writer.writerow(['File Name', 'Prediction'])
        for folder_name in os.listdir(output_dir):
            folder_path = os.path.join(parent_dir, folder_name)
            # os.makedirs(os.path.join(output_dir, folder_name))
            if os.path.isdir(folder_path):
                print(f"Folder Name: {folder_name}")

                for audio_file in os.listdir(folder_path):
                    audio_path = os.path.join(folder_path, audio_file)
                    print(audio_path)
                    img = cv2.imread(audio_path)
                    img = cv2.resize(img, (240, 320))
                    # Preprocess th audio file (e.g., compute spectrogram)
                    preprocessed_data = tf.cast(img, tf.float32) / 255.0
                    print(img.shape)

                    # Create a NumPy array or tensor with the preprocessed data
                    input_data = np.expand_dims(preprocessed_data, axis=0)

                    # Perform inference
                    single_prediction = cnn_model.predict(input_data)
                    print(single_prediction)
                    predictions = np.argmax(single_prediction, axis=1)
                    #print(dict[str(int(predictions))])
                    file_name = audio_file.replace('.png', '')

                    # Write the file name and prediction to the CSV file
                    string1 = file_name + "," + str(dict[str(int(predictions))])
                    writer.writerow([string1])





