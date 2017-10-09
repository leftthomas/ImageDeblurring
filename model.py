from keras.layers import Input, concatenate
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# the paper defined hyper-parameter:chr
channel_rate = 64


# Dense Block
def dense_block(inputs, dilation_factor=None):
    x = LeakyReLU(alpha=0.2)(inputs)
    x = Convolution2D(filters=4 * channel_rate, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    # the 3 × 3 convolutions along the dense field are alternated between ‘spatial’ convolution
    # and ‘dilated’ convolution with linearly increasing dilation factor
    if dilation_factor is not None:
        x = Convolution2D(filters=channel_rate, kernel_size=(3, 3), padding='same', dilation_rate=dilation_factor)(x)
    else:
        x = Convolution2D(filters=channel_rate, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    # add Gaussian noise
    x = Dropout(rate=0.5)(x)
    return x


def generator_model():
    # Input Image
    inputs = Input(shape=(256, 256, 3))
    # The Head
    h = Convolution2D(filters=4 * channel_rate, kernel_size=(3, 3), padding='same')(inputs)

    # The Dense Field
    d_1 = dense_block(inputs=h)
    x = concatenate([h, d_1])
    # the paper used dilated convolution at every even numbered layer within the dense field
    d_2 = dense_block(inputs=x, dilation_factor=(1, 1))
    x = concatenate([x, d_2])
    d_3 = dense_block(inputs=x)
    x = concatenate([x, d_3])
    d_4 = dense_block(inputs=x, dilation_factor=(2, 2))
    x = concatenate([x, d_4])
    d_5 = dense_block(inputs=x)
    x = concatenate([x, d_5])
    d_6 = dense_block(inputs=x, dilation_factor=(3, 3))
    x = concatenate([x, d_6])
    d_7 = dense_block(inputs=x)
    x = concatenate([x, d_7])
    d_8 = dense_block(inputs=x, dilation_factor=(2, 2))
    x = concatenate([x, d_8])
    d_9 = dense_block(inputs=x)
    x = concatenate([x, d_9])
    d_10 = dense_block(inputs=x, dilation_factor=(1, 1))
    # The Tail
    x = LeakyReLU(alpha=0.2)(d_10)
    x = Convolution2D(filters=4 * channel_rate, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    # The Global Skip Connection
    x = concatenate([h, x])
    x = Convolution2D(filters=channel_rate, kernel_size=(3, 3), padding='same')(x)
    x = PReLU()(x)

    # Output Image
    outputs = Convolution2D(filters=3, kernel_size=(3, 3), padding='same', activation='tanh')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# g = generator_model()
# print(g.summary())


def discriminator_model():
    # PatchGAN
    inputs = Input(shape=(channel_rate, channel_rate, 3))
    x = Convolution2D(filters=channel_rate, kernel_size=(3, 3), strides=(2, 2), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Convolution2D(filters=2 * channel_rate, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Convolution2D(filters=4 * channel_rate, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Convolution2D(filters=4 * channel_rate, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)

    # discriminator
    inputs = [Input(shape=(channel_rate, channel_rate, 3)) for _ in range(16)]
    x = [model(patch) for patch in inputs]
    outputs = concatenate(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# d = discriminator_model()
# print(d.summary())


def generator_containing_discriminator(image_shape, generator, discriminator):
    inputs = Input(image_shape)
    x_generator = generator(inputs)
    # Note the inputs first, then generated samples
    merged = concatenate([inputs, x_generator])
    # fixed d to train generator
    discriminator.trainable = False
    x_discriminator = discriminator(merged)
    model = Model(inputs, [x_generator, x_discriminator])
    return model
