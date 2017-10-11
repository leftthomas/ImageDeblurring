import keras.backend as K
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model

import data_utils

# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 3)
K_1 = 145
K_2 = 170


def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    # let the loss model can't be trained
    loss_model.trainable = False
    # loss_model.summary()
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def generator_loss(y_true, y_pred):
    return K_1 * perceptual_loss(y_true, y_pred) + K_2 * l1_loss(y_true, y_pred)


def adversarial_loss(y_true, y_pred):
    return -K.log(y_pred)


if __name__ == '__main__':
    a, b = data_utils.format_image('data/small/test/301.jpg', size=256)
    print(l1_loss(a.astype(np.float32), b.astype(np.float32)))
