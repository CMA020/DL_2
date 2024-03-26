from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input, DepthwiseConv2D, GlobalAveragePooling2D, Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization
from keras.layers import Flatten, Dense
from keras.callbacks import Callback
import tensorflow as tf
import gc

#from tensorflow.python.tpu import tpu_function
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

mish = Lambda(lambda x: x * K.tanh(K.softplus(x)))
physical_devices = tf.config.list_physical_devices('GPU')
import random
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(320, 240, 3))))
        labels.append(label)

    return images, labels
class ClearMemoryCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

if __name__ == '__main__':
    parent_dir = os.path.expanduser("/content/drive/MyDrive/Train")
    parent_dir2 = os.path.expanduser("/content/drive/MyDrive/Test")
    tf.keras.backend.clear_session()
    #pretrained_model = load_model(os.path.expanduser("/content/model_weights.h5"))

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for i, folder_name in enumerate(os.listdir(parent_dir)):

        folder_path = os.path.join(parent_dir, folder_name)
        images, labels = load_images_from_path(folder_path, i)
        num_images = len(images)
        quarter_len = num_images // 4

# Get a random sample of indices for the quarter of data
        random_indices = random.sample(range(num_images), quarter_len)

        # Use the random indices to select a quarter of the data
        train_images.extend([images[i] for i in random_indices])
        train_labels.extend([labels[i] for i in random_indices])

        folder_path2 = os.path.join(parent_dir2, folder_name)

        images2, labels2 = load_images_from_path(folder_path2, i)

        num_images2 = len(images2)
        quarter_len2 = num_images2 // 4

# Get a random sample of indices for the quarter of data
        random_indices2 = random.sample(range(num_images2), quarter_len2)
        test_images.extend([images[i] for i in random_indices2])
        test_labels.extend([labels[i] for i in random_indices2])
        gc.collect()

        print(i, ":", folder_name)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    train_labels_encoded = to_categorical(train_labels, 13)
    test_labels_encoded = to_categorical(test_labels, 13)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels_encoded))
    train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_images)).batch(6)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels_encoded))
    test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(6)

    gc.collect()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=5)


    #tpu_function.get_tpu_context().set_config(experimental_run_tf_rewrite=True)
    with tf.device('/device:GPU:0'):
      model = efficientnet((320,240,3),13)
      model.load_weights(os.path.expanduser("/content/drive/MyDrive/model_weights_e_5.h5"))

      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
      hist = model.fit(train_dataset, validation_data=test_dataset, epochs=18,callbacks=[reduce_lr])




      current_dir = os.getcwd()
      weights_file = os.path.join(current_dir, 'model_weights_e_6.h5')
      model.save_weights(weights_file)

      acc = hist.history['accuracy']
      val_acc = hist.history['val_accuracy']
      epochs = range(1, len(acc) + 1)

      plt.plot(epochs, acc, '-', label='Training Accuracy')
      plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend(loc='lower right')
      plt.show()