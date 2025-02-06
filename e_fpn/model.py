
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import SeparableConv2D, Conv2D, MaxPooling2D, concatenate

# Activation function
activation_ = tf.keras.activations.relu

def conv_block(inputs, filters, kernel_size=(3, 3), padding='same', activation=activation_):
    conv = layers.SeparableConv2D(filters, kernel_size, padding=padding)(inputs)
    conv = layers.BatchNormalization()(conv)
    conv = activation(conv)
    return conv  

def inception_sepconvblock(inputs, f1, f2, f3):
    conva = SeparableConv2D(f1, (3, 3), activation='relu', padding='same')(inputs)
    conva = SeparableConv2D(f1, (3, 3), activation='relu', padding='same')(conva)

    convb = SeparableConv2D(f2, (5, 5), activation='relu', padding='same')(inputs)
    convb = SeparableConv2D(f2, (5, 5), activation='relu', padding='same')(convb)

    convc = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    convc = Conv2D(f3, (1, 1), padding='same')(convc)

    concatenated = concatenate([conva, convb, convc])
    return concatenated  

def build_fpn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Bottom-up pathway
    conv1 = inception_sepconvblock(inputs, f1=22, f2=21, f3=21)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = inception_sepconvblock(pool1, f1=44, f2=42, f3=42)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = inception_sepconvblock(pool2, f1=86, f2=85, f3=85)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = inception_sepconvblock(pool3, f1=171, f2=171, f3=170)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = inception_sepconvblock(pool4, f1=342, f2=341, f3=341)

    # Top-down pathway
    p5 = conv_block(conv5, 128, kernel_size=(1, 1))  
    p4 = conv_block(conv4, 128, kernel_size=(1, 1)) + tf.keras.layers.UpSampling2D(size=(2, 2))(p5)
    p3 = conv_block(conv3, 128, kernel_size=(1, 1)) + tf.keras.layers.UpSampling2D(size=(2, 2))(p4)  
    p2 = conv_block(conv2, 128, kernel_size=(1, 1)) + tf.keras.layers.UpSampling2D(size=(2, 2))(p3)  

    # Prediction heads
    output_p3 = layers.SeparableConv2D(num_classes, (3, 3), padding='same', activation='softmax')(p3)
    output_p4 = layers.SeparableConv2D(num_classes, (3, 3), padding='same', activation='softmax')(p4)
    output_p5 = layers.SeparableConv2D(num_classes, (3, 3), padding='same', activation='softmax')(p5)
    output_p2 = layers.SeparableConv2D(num_classes, (3, 3), padding='same', activation='softmax')(p2)

    # Final prediction merging
    output1 = (
        tf.keras.layers.UpSampling2D(size=(2, 2))(output_p2) +
        tf.keras.layers.UpSampling2D(size=(4, 4))(output_p3) +
        tf.keras.layers.UpSampling2D(size=(8, 8))(output_p4) +
        tf.keras.layers.UpSampling2D(size=(16, 16))(output_p5)
    )
    output = layers.SeparableConv2D(num_classes, (3, 3), padding='same', activation='softmax')(output1)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model
