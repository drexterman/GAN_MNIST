import tensorflow as tf
from .generator import build_generator
from .discriminator import build_discriminator

def build_gan(latent_dim, img_shape):
    # Build generator and discriminator
    generator = build_generator(latent_dim)
    discriminator = build_discriminator(img_shape)

    # Compile discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Combined GAN model
    discriminator.trainable = False
    noise_input = tf.keras.Input(shape=(latent_dim,))
    slider_input = tf.keras.Input(shape=(1,))
    img = generator([noise_input, slider_input])
    validity = discriminator([img, slider_input])
    combined = tf.keras.Model([noise_input, slider_input], validity)
    combined.compile(loss='binary_crossentropy', optimizer='adam')

    return generator, discriminator, combined