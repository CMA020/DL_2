import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import os
import gc
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings('ignore')

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
    y, sr = librosa.load(audio_path)

    # Create the spectrogram
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()

    # Save the spectrogram as an image
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    input_extension = os.path.splitext(os.path.basename(audio_path))[1]
    output_filename = f'{filename}{input_extension}.png'
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close()

gc.collect()


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
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        os.makedirs(os.path.join(output_dir, folder_name))
        if os.path.isdir(folder_path):
            print(f"Folder Name: {folder_name}")

            for audio_file in os.listdir(folder_path):
                if audio_file.endswith('.wav') or audio_file.endswith('.mp3') or audio_file.endswith(
                        '.aiff') or audio_file.endswith('.flac') or audio_file.endswith('.ogg') or audio_file.endswith(
                    '.m4a'):
                    audio_path = os.path.join(folder_path, audio_file)
                    generate_spectrogram(audio_path, os.path.join(output_dir, folder_name))

   
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
                    predictions = np.argmax(single_prediction, axis=1)
                    print(dict[str(int(predictions))])
                    file_name = audio_file.replace('.png', '')

                    # Write the file name and prediction to the CSV file
                    string1 = file_name + "," + str(dict[str(int(predictions))])
                    writer.writerow([string1])





