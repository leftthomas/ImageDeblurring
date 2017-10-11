import numpy as np

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

    d.trainable = True
    d.compile(optimizer='adam', loss='binary_crossentropy')

    # # 下面在满足epoch条件下进行训练
    # for epoch in range(epoch_num):
    #     print("Epoch is", epoch)
    #
    #     # 计算一个epoch所需要的迭代数量，即训练样本数除批量大小数的值取整；其中shape[0]就是读取矩阵第一维度的长度
    #     print("Number of batches", int(x_train.shape[0] / batch_size))
    #
    #     # 在一个epoch内进行迭代训练
    #     for index in range(int(x_train.shape[0] / batch_size)):
    #
    #         # 随机生成的噪声服从均匀分布，且采样下界为-1、采样上界为1，输出BATCH_SIZE×100个样本；即抽取一个批量的随机样本
    #         noise = np.random.uniform(-1, 1, size=(batch_size, 100))
    #
    #         # 抽取一个批量的真实图片
    #         image_batch = x_train[index * batch_size:(index + 1) * batch_size]
    #
    #         # 生成的图片使用生成器对随机噪声进行推断；verbose为日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录
    #         generated_images = g.predict(noise, verbose=0)
    #
    #         # 每经过100次迭代输出一张生成的图片
    #         if index % 100 == 0:
    #             image = combine_images(generated_images)
    #             image = image * 127.5 + 127.5
    #             Image.fromarray(image.astype(np.uint8)).save("GAN/" + str(epoch) + "_" + str(index) + ".png")
    #
    #         # 将真实的图片和生成的图片以多维数组的形式拼接在一起，真实图片在上，生成图片在下
    #         x = np.concatenate((image_batch, generated_images))
    #
    #         # 生成图片真假标签，即一个包含两倍批量大小的列表；前一个批量大小都是1，代表真实图片，后一个批量大小都是0，代表伪造图片
    #         y = [1] * batch_size + [0] * batch_size
    #
    #         # 判别器的损失；在一个batch的数据上进行一次参数更新
    #         d_loss = d.train_on_batch(x, y)
    #         print("batch %d d_loss : %f" % (index, d_loss))
    #
    #         # 随机生成的噪声服从均匀分布
    #         noise = np.random.uniform(-1, 1, (batch_size, 100))
    #
    #         # 固定判别器
    #         d.trainable = False
    #
    #         # 计算生成器损失；在一个batch的数据上进行一次参数更新
    #         g_loss = d_on_g.train_on_batch(noise, [1] * batch_size)
    #
    #         # 令判别器可训练
    #         d.trainable = True
    #         print("batch %d g_loss : %f" % (index, g_loss))
    #
    #         # 每100次迭代保存一次生成器和判别器的权重
    #         if index % 100 == 9:
    #             g.save_weights('generator_weights.h5', True)
    #             d.save_weights('discriminator_weights.h5', True)


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
