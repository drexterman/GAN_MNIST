import numpy as np
import tensorflow as tf

def load_mnist():
    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255.0  # Normalize to [0, 1]
    X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
    return X_train, y_train