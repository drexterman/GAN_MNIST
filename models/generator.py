from tensorflow.keras import layers
import tensorflow as tf

def build_generator(latent_dim):
    # Noise input
    noise_input = layers.Input(shape=(latent_dim,))
    # Slider input (continuous value between 0 and 9)
    slider_input = layers.Input(shape=(1,))

    # Embed the slider input
    slider_embedding = layers.Dense(128)(slider_input)
    slider_embedding = layers.LeakyReLU(alpha=0.2)(slider_embedding)

    # Concatenate noise and slider embedding
    combined_input = layers.Concatenate()([noise_input, slider_embedding])

    # Generator network
    x = layers.Dense(128 * 7 * 7)(combined_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((7, 7, 128))(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same')(x)

    # Define the model
    model = tf.keras.Model([noise_input, slider_input], x)
    return model