# You are free to either implement both test() and evaluate() function, or implement test_batch() and evaluate_batch() function. Apart from the 2 functions which you must mandatorily implement, you are free to implement some helper functions as per your convenience.

# Import all necessary python libraries here
# Do not write import statements anywhere else
import os
import pandas as pd
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
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input, DepthwiseConv2D, GlobalAveragePooling2D, Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D,AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
import csv
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

TEST_DATA_DIRECTORY_ABSOLUTE_PATH = os.path.expanduser("~/DL_1/audio_dataset/val/car_horn")   ###################
OUTPUT_CSV_ABSOLUTE_PATH = os.path.expanduser('~/DL_1/1.csv')     ###############

def depthwise_block(x, filters, kernel_size, strides, expansion_factor, block_id):
    # Depthwise convolution
    x = DepthwiseConv2D(kernel_size, strides=strides, padding='same', depth_multiplier=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    pointwise_conv_filters = max(1, int(filters * expansion_factor))
    x = Conv2D(pointwise_conv_filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def efficientnet(input_shape, num_classes):

    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = depthwise_block(x, 16, (3, 3), strides=(1, 1), expansion_factor=1, block_id=0)

    x = depthwise_block(x, 24, (3, 3), strides=(2, 2), expansion_factor=6, block_id=1)
    x = depthwise_block(x, 24, (3, 3), strides=(1, 1), expansion_factor=6, block_id=2)

    x = depthwise_block(x, 40, (5, 5), strides=(2, 2), expansion_factor=6, block_id=3)
    x = depthwise_block(x, 40, (5, 5), strides=(1, 1), expansion_factor=6, block_id=4)

    x = depthwise_block(x, 80, (3, 3), strides=(2, 2), expansion_factor=6, block_id=5)
    x = depthwise_block(x, 80, (3, 3), strides=(1, 1), expansion_factor=6, block_id=6)

    x = depthwise_block(x, 112, (5, 5), strides=(1, 1), expansion_factor=6, block_id=7)
    x = depthwise_block(x, 112, (5, 5), strides=(1, 1), expansion_factor=6, block_id=8)

    x = depthwise_block(x, 192, (5, 5), strides=(2, 2), expansion_factor=6, block_id=9)
    x = depthwise_block(x, 192, (5, 5), strides=(1, 1), expansion_factor=6, block_id=10)

    x = depthwise_block(x, 320, (3, 3), strides=(1, 1), expansion_factor=6, block_id=11)

    x = Conv2D(1280, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model
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




dict={'0': 12, '1': 3 , '2':11,'3':13 , '4':4 , '5':5 , '6':1 , '7':7 , '8':10 , '9':2 , '10':8 , '11':6 , '12':9  }



# The above two variables will be changed during testing. The current values are an example of what their contents would look like.

def evaluate(file_path):
    global output_dir2,output_dir
    predicted_class=None
    # Write your code to predict class for a single audio file instance here
    cnn_model = efficientnet((320,240,3), 13)
    current_dir = os.getcwd()
    cnn_model.load_weights(os.path.expanduser(os.path.join(current_dir, "model_weights_e_5.h5")))
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    output_dir2 = os.path.expanduser(os.path.join(current_dir, "Test"))
    os.makedirs(output_dir2)

    output_dir = os.path.expanduser(os.path.join(current_dir, "Test2"))
    os.makedirs(output_dir)
    input_file = file_path
    filename = file_path.split('/')[-1]
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".wav":
        output_file = os.path.join(output_dir2, filename+".wav")
        shutil.copy(input_file, output_file)

    elif ext in [".mp3", ".aiff",'.flac','.ogg','.m4a']:
        output_file = os.path.join(output_dir2,  filename + ".wav")
        convert_audio(input_file, output_file)
    for input_file in os.listdir(output_dir2):
        if input_file.endswith('.wav') or input_file.endswith('.mp3') or input_file.endswith(
                '.aiff') or input_file.endswith('.flac') or input_file.endswith('.ogg') or input_file.endswith(
            '.m4a'):
            #audio_path = os.path.join(output_dir2, input_file)
            generate_spectrogram(file_path, output_dir)

    for audio_file in os.listdir(output_dir):
        # os.makedirs(os.path.join(output_dir, folder_name))
        audio_path = os.path.join(output_dir, audio_file)
        print(audio_path)
        img = cv2.imread(audio_path)
        print(img.shape)
        img = cv2.resize(img, (240, 320))
        # Preprocess th audio file (e.g., compute spectrogram)
        preprocessed_data = tf.cast(img, tf.float32) / 255.0

        # Create a NumPy array or tensor with the preprocessed data
        input_data = np.expand_dims(preprocessed_data, axis=0)

        # Perform inference
        single_prediction = cnn_model.predict(input_data)
        predictions = np.argmax(single_prediction, axis=1)
        print(predictions)
        # print(single_prediction)

        file_name = audio_file.replace('.wav.png', '')

        # Write the file name and prediction to the CSV file
        string1 = file_name + "," + str(dict[str(int(predictions))])
        print(string1)
        predicted_class=dict[str(int(predictions))]
    shutil.rmtree(output_dir)
    shutil.rmtree(output_dir2)

    return predicted_class





def test():

    filenames = []
    predictions = []


    # for file_path in os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
    for file_name in os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
        # prediction = evaluate(file_path)
        absolute_file_name = os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, file_name)
        prediction = evaluate(absolute_file_name)

        filenames.append(absolute_file_name)
        predictions.append(prediction)
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)




# Uncomment exactly one of the two lines below, i.e. either execute test() or test_batch()
test()
