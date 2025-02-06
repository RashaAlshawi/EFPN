import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

def load_data():
    # Load the positive image and mask datasets
    X_P = np.load('//Users/rashaalshawi/Documents/Research_PhD23/E-FPN-Segmentation/dataset/PositiveImageNew.npy')
    y_P = np.load('/Users/rashaalshawi/Documents/Research_PhD23/E-FPN-Segmentation/dataset/PositivemaskNew.npy')

    # Shuffle the dataset to ensure randomness
    X_datas, y_datas = shuffle(X_P, y_P, random_state=10)

    return X_datas, y_datas

def normalize(input_image):
    # Normalize image to [0, 1] range
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

def prepare_data(X_datas, y_datas, num_classes=9, train_split=0.7, val_split=0.15):
    # Convert mask to categorical (one-hot encoding)
    train_masks_cat = to_categorical(y_datas, num_classes=num_classes)
    y_train_cat_test = train_masks_cat.reshape((y_datas.shape[0], y_datas.shape[1], y_datas.shape[2], num_classes))

    # Normalize the input images
    X_datas_ = normalize(X_datas)

    # Split data into training, validation, and test sets
    samples_train = int(train_split * len(X_datas_))
    samples_val = int(val_split * len(X_datas_))

    X_train = X_datas_[:samples_train]
    y_train = y_train_cat_test[:samples_train]

    X_val = X_datas_[samples_train:samples_train + samples_val]
    y_val = y_train_cat_test[samples_train:samples_train + samples_val]

    X_test = X_datas_[samples_train + samples_val:]
    y_test = y_train_cat_test[samples_train + samples_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test
