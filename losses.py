import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model

# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 3)


def l1_loss(y_true, y_pred):
    return K.mean(K.sum(K.abs(y_pred - y_true)))


# a,b=data_utils.format_image('data/small/test/301.jpg',size=256)
# print(l1_loss(a.astype(np.float32),b.astype(np.float32)))


def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    # let the parameters can't be trained
    for l in vgg.layers:
        l.trainable = False
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.summary()
    return K.mean(K.sum(K.square(loss_model(y_true) - loss_model(y_pred))))


def adversarial_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)
