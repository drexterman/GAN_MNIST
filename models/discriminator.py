from tensorflow.keras import layers
import numpy as np
import tensorflow as tf

def build_discriminator(img_shape):
    # Image input
    img_input = layers.Input(shape=img_shape)
    # Slider input (continuous value between 0 and 9)
    slider_input = layers.Input(shape=(1,))

    # Embed the slider input
    slider_embedding = layers.Dense(128)(slider_input)
    slider_embedding = layers.LeakyReLU(alpha=0.2)(slider_embedding)
    slider_embedding = layers.Dense(np.prod(img_shape))(slider_embedding)
    slider_embedding = layers.Reshape(img_shape)(slider_embedding)

    # Concatenate image and slider embedding
    combined_input = layers.Concatenate()([img_input, slider_embedding])

    # Discriminator network
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(combined_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    # Define the model
    model = tf.keras.Model([img_input, slider_input], x)
    return model