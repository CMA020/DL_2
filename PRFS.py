import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import os
import gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')
from tensorflow.keras.utils import to_categorical
import csv
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import random


def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(320, 240, 3))))
        labels.append(label)

    return images, labels


dict = {'0': 12, '1': 3, '2': 11, '3': 13, '4': 4, '5': 5, '6': 1, '7': 7, '8': 10, '9': 2, '10': 8, '11': 6, '12': 9}

if __name__ == '__main__':
    test_images = []
    test_labels = []

    parent_dir = os.path.expanduser("~/DL_1/Train")
    parent_dir2 = os.path.expanduser("~/DL_1/Test")
    # os.makedirs("~/DL_1/Test2")
    #output_dir = os.path.expanduser("~/DL_1/Test2")
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
    model.load_weights(os.path.expanduser("/home/cma/DL_2/DL_2/model_weights11.h5"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    for i, folder_name in enumerate(os.listdir(parent_dir)):
        folder_path = os.path.join(parent_dir2, folder_name)
        images, labels = load_images_from_path(folder_path, i)
        num_images = len(images)
        quarter_len = num_images // 1

        # Get a random sample of indices for the quarter of data
        random_indices = random.sample(range(num_images), quarter_len)

        # Use the random indices to select a quarter of the data
        # test_images.extend([images[i] for i in random_indices])
        # test_labels.extend([labels[i] for i in random_indices])
        test_images.extend(images)
        test_labels.extend(labels)
        print(i, ":", folder_name)
        gc.collect()
    test_labels_encoded = to_categorical(test_labels, 13)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels_encoded))
    test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(6)
    gc.collect()

    val_loss, val_accuracy = model.evaluate(test_dataset, verbose=0)

    y_pred = np.argmax(model.predict(test_dataset), axis=1)

    # Get true labels
    y_true = np.argmax(test_labels_encoded, axis=1)

    # Compute precision, recall, F1-score, and support for each class
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    print("Class\tPrecision\tRecall\t\tF1-Score\tSupport")
    print("-----\t---------\t------\t\t--------\t-------")
    for i in range(13):
        print(f"{i}\t{precision[i]:.3f}\t\t{recall[i]:.3f}\t\t{f1_score[i]:.3f}\t\t{support[i]}")

    # val_images = val_generator.filenames
    # val_img = np.asarray(val_images)[errors]






