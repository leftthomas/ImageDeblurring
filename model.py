from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential

# the paper defined hyper-parameter:chr
channel_rate = 64


def generator_model():
    # Input Image
    inputs = Input(shape=(256, 256, 3))
    # The Head
    h = Convolution2D(filters=4 * channel_rate, kernel_size=(3, 3), padding='same')(inputs)

    # The Dense Field
    d_1 = dense_block(inputs=h)
    x = Concatenate([h, d_1])
    # the paper used dilated convolution at every even numbered layer within the dense field
    d_2 = dense_block(inputs=x, dilation_factor=(1, 1))
    x = Concatenate([x, d_2])
    d_3 = dense_block(inputs=x)
    x = Concatenate([x, d_3])
    d_4 = dense_block(inputs=x, dilation_factor=(2, 2))
    x = Concatenate([x, d_4])
    d_5 = dense_block(inputs=x)
    x = Concatenate([x, d_5])
    d_6 = dense_block(inputs=x, dilation_factor=(3, 3))
    x = Concatenate([x, d_6])
    d_7 = dense_block(inputs=x)
    x = Concatenate([x, d_7])
    d_8 = dense_block(inputs=x, dilation_factor=(2, 2))
    x = Concatenate([x, d_8])
    d_9 = dense_block(inputs=x)
    x = Concatenate([x, d_9])
    d_10 = dense_block(inputs=x, dilation_factor=(1, 1))
    # The Tail
    x = LeakyReLU(alpha=0.2)(d_10)
    x = Convolution2D(filters=4 * channel_rate, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    # The Global Skip Connection
    x = Concatenate([h, x])
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


m = generator_model()
print(m.summary())


def discriminator_model():
    # Note the input channel is 6
    inputs = Input(shape=(256, 256, 3 * 2))
    d = ZeroPadding2D(padding=(1, 1))(inputs)
    d = Convolution2D(filters=48, kernel_size=(4, 4), strides=(2, 2))(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = ZeroPadding2D(padding=(1, 1))(d)
    d = Convolution2D(filters=48 * 2, kernel_size=(4, 4), strides=(2, 2))(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = ZeroPadding2D(padding=(1, 1))(d)
    d = Convolution2D(filters=48 * 4, kernel_size=(4, 4), strides=(2, 2))(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = ZeroPadding2D(padding=(1, 1))(d)
    d = Convolution2D(filters=48 * 8, kernel_size=(4, 4), strides=(2, 2))(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = ZeroPadding2D(padding=(1, 1))(d)
    outputs = Convolution2D(filters=1, kernel_size=(4, 4), strides=(1, 1), activation='sigmoid')(d)
    model = Model(inputs, outputs)
    return model


def generator_containing_discriminator(g, d):
    # 将前面定义的生成器架构和判别器架构组拼接成一个大的神经网络，用于判别生成的图片
    model = Sequential()
    # 先添加生成器架构，再令d不可训练，即固定d
    # 因此在给定d的情况下训练生成器，即通过将生成的结果投入到判别器进行辨别而优化生成器
    model.add(g)
    d.trainable = False
    model.add(d)
    return model
