from keras.layers import Input, concatenate
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# the paper defined hyper-parameter:chr
channel_rate = 64


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
    ############### Global Skip这里的输出维度作者设了多少不确定 ###############
    x = Convolution2D(filters=channel_rate, kernel_size=(3, 3), padding='same')(x)
    ############### Global Skip这里的输出维度作者设了多少不确定 ###############
    x = PReLU()(x)

    # Output Image
    outputs = Convolution2D(filters=3, kernel_size=(3, 3), padding='same')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


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
    ############### Dropout这里不确定作者用的是不是这个 ###############
    x = Dropout(rate=0.5)(x)
    ############### Dropout这里不确定作者用的是不是这个 ###############
    return x


# g = generator_model()
# print(g.summary())


def discriminator_model():
    # Note the input channel is 6
    inputs = Input(shape=(256, 256, 3 * 2))
    x = ZeroPadding2D(padding=(1, 1))(inputs)
    x = Convolution2D(filters=48, kernel_size=(4, 4), strides=(2, 2))(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(filters=48 * 2, kernel_size=(4, 4), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(filters=48 * 4, kernel_size=(4, 4), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(filters=48 * 8, kernel_size=(4, 4), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    outputs = Convolution2D(filters=1, kernel_size=(4, 4), strides=(1, 1), activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model


# d = discriminator_model()
# print(d.summary())


def generator_containing_discriminator(generator, discriminator):
    inputs = Input((256, 256, 3))
    x_generator = generator(inputs)
    # Note the inputs first, then generated samples
    merged = concatenate([inputs, x_generator])
    # fixed d
    discriminator.trainable = False
    x_discriminator = discriminator(merged)
    model = Model(inputs, [x_generator, x_discriminator])
    return model
