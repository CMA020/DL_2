import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
from pydub import AudioSegment
import os
import gc

#s.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings('ignore')
import shutil
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D,AveragePooling2D
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
        Conv2D(32, (3, 3), activation='silu', input_shape=(320, 240, 3)),
        AveragePooling2D(2, 2),
        Conv2D(128, (3, 3), activation='silu'),
        AveragePooling2D(2, 2),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation='silu'),
        BatchNormalization(),
        AveragePooling2D(2, 2),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation='silu'),
        BatchNormalization(),
        AveragePooling2D(2, 2),
        Conv2D(128, (3, 3), activation='silu'),
        Dropout(0.3),
        AveragePooling2D(2, 2),
        Flatten(),
        Dropout(0.3),
        Dense(1024, activation='silu'),
        BatchNormalization(),
        Dense(13, activation='softmax')
    ])


    return model

dict={'0': 12, '1': 3 , '2':11,'3':13 , '4':4 , '5':5 , '6':1 , '7':7 , '8':10 , '9':2 , '10':8 , '11':6 , '12':9  }
if __name__ == '__main__':
    cnn_model = create_model()
    current_dir = os.getcwd()
    cnn_model.load_weights(os.path.expanduser(os.path.join(current_dir,"model_weights11.h5")))
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    parent_dir = os.path.expanduser('~/DL_1/audio_dataset/val/Fart')  ##### add  absolutepatth (test directory)
    output_dir2=os.path.expanduser(os.path.join(current_dir,"Test"))
    os.makedirs(output_dir2)
    for filename  in os.listdir(parent_dir):
        input_file = os.path.join(parent_dir,  filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".wav":
            output_file = os.path.join(output_dir2, filename+".wav")
            shutil.copy(input_file, output_file)

        elif ext in [".mp3", ".aiff"]:
            output_file = os.path.join(output_dir2,  filename + ".wav")
            convert_audio(input_file, output_file)

    os.makedirs(os.path.join(current_dir,"Test2"))
    output_dir = os.path.expanduser(os.path.join(current_dir,"Test2"))

    for audio_file in os.listdir(output_dir2):

        if audio_file.endswith('.wav') or audio_file.endswith('.mp3') or audio_file.endswith(
                '.aiff') or audio_file.endswith('.flac') or audio_file.endswith('.ogg') or audio_file.endswith(
            '.m4a'):
            audio_path = os.path.join(output_dir2, audio_file)
            generate_spectrogram(audio_path, output_dir)
    with open(os.path.expanduser('~/DL_1/1.csv'), mode='w', newline='') as file:    ###W add csv file path (absolute)
        writer = csv.writer(file)

        # Write header row
        writer.writerow(['File Name', 'Prediction'])
        for audio_file in os.listdir(output_dir):

            # os.makedirs(os.path.join(output_dir, folder_name))
            audio_path = os.path.join(output_dir, audio_file)
            print(audio_path)
            img = cv2.imread(audio_path)
            img = cv2.resize(img, (240, 320))
            # Preprocess th audio file (e.g., compute spectrogram)
            preprocessed_data = tf.cast(img, tf.float32) / 255.0


            # Create a NumPy array or tensor with the preprocessed data
            input_data = np.expand_dims(preprocessed_data, axis=0)

            # Perform inference
            single_prediction = cnn_model.predict(input_data)
            predictions = np.argmax(single_prediction, axis=1)
            print(predictions)
            #print(single_prediction)

            file_name = audio_file.replace('.wav.png', '')

            # Write the file name and prediction to the CSV file
            string1 = file_name + "," + str(dict[str(int(predictions))])
            writer.writerow([string1])






