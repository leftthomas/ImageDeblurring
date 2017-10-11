import numpy as np
from PIL import Image

import data_utils
from model import generator_model, discriminator_model, generator_containing_discriminator


def train(batch_size, epoch_num):
    # Note the x(blur) in the second, the y(full) in the first
    y_train, x_train = data_utils.load_data(data_type='train')

    # GAN
    g = generator_model()
    d = discriminator_model()
    d_on_g = generator_containing_discriminator(g, d)

    # compile the generator model, use default optimizer parameters
    g.compile(optimizer='adam', loss='binary_crossentropy')
    d.compile(optimizer='adam', loss='binary_crossentropy')
    d_on_g.compile(optimizer='adam', loss='binary_crossentropy')

    for epoch in range(epoch_num):
        print("Epoch is", epoch + 1)
        print("Number of batches", int(x_train.shape[0] / batch_size))

        for index in range(int(x_train.shape[0] / batch_size)):
            # 抽取一个批量的数据
            image_blur_batch = x_train[index * batch_size:(index + 1) * batch_size]
            image_full_batch = y_train[index * batch_size:(index + 1) * batch_size]
            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

            # 每经过100次迭代输出一张生成的图片
            if index % 100 == 0:
                image = generated_images * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save("result/" + str(epoch + 1) + "_" + str(index + 1) + ".png")

            # 将真实的图片和生成的图片以多维数组的形式拼接在一起，真实图片在上，生成图片在下
            x = np.concatenate((image_full_batch, generated_images))

            # 生成图片真假标签，即一个包含两倍批量大小的列表；前一个批量标签都是1，代表真实图片，后一个批量标签都是0，代表伪造图片
            y = [1] * batch_size + [0] * batch_size

            # 判别器的损失；在一个batch的数据上进行一次参数更新
            d_loss = d.train_on_batch(x, y)
            print("batch %d d_loss : %f" % (index + 1, d_loss))

            # 固定判别器
            d.trainable = False

            # 计算生成器损失；在一个batch的数据上进行一次参数更新
            g_loss = d_on_g.train_on_batch(image_blur_batch, [1] * batch_size)

            # 令判别器可训练
            d.trainable = True
            print("batch %d g_loss : %f" % (index + 1, g_loss))

            # 每100次迭代保存一次生成器和判别器的权重
            if index % 100 == 9:
                g.save_weights('weight/generator_weights.h5', True)
                d.save_weights('weight/discriminator_weights.h5', True)


def test(batch_size):
    # Note the x(blur) in the second, the y(full) in the first
    y_test, x_test = data_utils.load_data(data_type='test')
    # 训练完模型后，可以运行该函数生成图片
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator_weights.h5')
    noise = np.random.uniform(-1, 1, (batch_size, 100))
    generated_images = g.predict(noise, verbose=0)


# train(132)
# test(132)
