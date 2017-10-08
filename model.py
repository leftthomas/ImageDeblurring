from keras.layers import Input, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten
from keras.layers.noise import GaussianDropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential

# the paper defined hyper-parameter:chr
channel_rate = 64


def generator_model():
    # Input Image
    inputs = Input(shape=(256, 256, 3))
    # The Head
    x = Conv2D(filters=4 * channel_rate, kernel_size=(3, 3), padding='same')(inputs)
    # The Dense Field

    model = Model(inputs=inputs, outputs=x)
    return model


# Dense Block
def dense_block(inputs, dilation_factor=None):
    x = LeakyReLU()(inputs)
    x = Conv2D(filters=4 * channel_rate, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # the paper used dilated convolution at every even numbered layer within the dense field
    if dilation_factor is not None:
        x = Conv2D(filters=channel_rate, kernel_size=(3, 3), padding='same', dilation_rate=dilation_factor)(x)
    else:
        x = Conv2D(filters=channel_rate, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    # add Gaussian noise
    x = GaussianDropout(rate=0.5)(x)
    return x


m = generator_model()
print(m.output)


def discriminator_model():
    # 下面搭建判别器架构，同样采用序贯模型
    model = Sequential()
    # 添加2维卷积层，卷积核大小为5×5，激活函数为tanh，输出为64维
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    # 为空域信号施加最大值池化，pool_size取（2，2）代表使图片在两个维度上均变为原长的一半
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten层把多维输入一维化，常用在从卷积层到全连接层的过渡
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    # 一个结点进行二值分类，并采用sigmoid函数的输出作为概念
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
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
