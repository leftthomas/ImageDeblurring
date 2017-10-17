from keras.layers import Input, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout, Dense, Flatten, Lambda
from keras.layers.merge import Average
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils.vis_utils import plot_model

# the paper defined hyper-parameter:chr
channel_rate = 64
# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 3)
patch_shape = (channel_rate, channel_rate, 3)


# Dense Block
def dense_block(inputs, dilation_factor=None):
    x = LeakyReLU(alpha=0.2)(inputs)
    x = Convolution2D(filters=4 * channel_rate, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    # the 3 × 3 convolutions along the dense field are alternated between ‘spatial’ convolution
    # and ‘dilated’ convolution with linearly increasing dilation factor
    if dilation_factor is not None:
        x = Convolution2D(filters=channel_rate, kernel_size=(3, 3), padding='same',
                          dilation_rate=dilation_factor)(x)
    else:
        x = Convolution2D(filters=channel_rate, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    # add Gaussian noise
    x = Dropout(rate=0.5)(x)
    return x


def generator_model():
    # Input Image, Note the shape is variable
    inputs = Input(shape=(None, None, 3))
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
    # PReLU can't be used, because it is connected with the input shape
    # x = PReLU()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Output Image
    outputs = Convolution2D(filters=3, kernel_size=(3, 3), padding='same', activation='tanh')(x)
    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    return model


def discriminator_model():
    # PatchGAN
    inputs = Input(shape=patch_shape)
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
    model = Model(inputs=inputs, outputs=outputs, name='PatchGAN')
    # model.summary()

    # discriminator
    inputs = Input(shape=image_shape)

    list_row_idx = [(i * channel_rate, (i + 1) * channel_rate) for i in
                    range(int(image_shape[0] / patch_shape[0]))]
    list_col_idx = [(i * channel_rate, (i + 1) * channel_rate) for i in
                    range(int(image_shape[1] / patch_shape[1]))]

    list_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(inputs)
            list_patch.append(x_patch)

    x = [model(patch) for patch in list_patch]
    outputs = Average()(x)
    model = Model(inputs=inputs, outputs=outputs, name='Discriminator')
    return model


def generator_containing_discriminator(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    g = generator_model()
    g.summary()
    d = discriminator_model()
    d.summary()
    plot_model(d)
    m = generator_containing_discriminator(generator_model(), discriminator_model())
    m.summary()
