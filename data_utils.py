import glob as gb
import os

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


# normalization x to [-1,1]
def normalization(x):
    return x / 127.5 - 1


# according the image path to read the image and covert it
# to the given size, then slice it, finally return the full and blur images
def format_image(image_path, size):
    image = Image.open(image_path)
    # slice image into full and blur images
    image_full = image.crop((0, 0, image.size[0] / 2, image.size[1]))
    # Note the full image in left, the blur image in right
    image_blur = image.crop((image.size[0] / 2, 0, image.size[0], image.size[1]))

    # image_full.show()
    # image_blur.show()

    image_full = image_full.resize((size, size), Image.ANTIALIAS)
    image_blur = image_blur.resize((size, size), Image.ANTIALIAS)

    # return the numpy arrays
    return np.array(image_full), np.array(image_blur)


# convert images to hdf5 data
def build_hdf5(jpeg_dir, size=256):
    # put data in HDF5
    hdf5_file = os.path.join('data', 'data.h5')
    with h5py.File(hdf5_file, 'w') as f:

        for data_type in tqdm(['train', 'test'], desc='create HDF5 dataset from images'):
            data_path = jpeg_dir + '/%s/*.jpg' % data_type
            images_path = gb.glob(data_path)
            # print(images_path)
            data_full = []
            data_blur = []
            for image_path in images_path:
                image_full, image_blur = format_image(image_path, size)
                data_full.append(image_full)
                data_blur.append(image_blur)

            # print(len(data_full))
            # print(len(data_blur))
            f.create_dataset('%s_data_full' % data_type, data=data_full)
            f.create_dataset('%s_data_blur' % data_type, data=data_blur)


# load data by data type
def load_data(data_type):
    with h5py.File('data/data.h5', 'r') as f:
        data_full = f['%s_data_full' % data_type][:].astype(np.float32)
        data_full = normalization(data_full)

        data_blur = f['%s_data_blur' % data_type][:].astype(np.float32)
        data_blur = normalization(data_blur)

        return data_full, data_blur


def generate_image(full, blur, generated, path, epoch=None, index=None):
    full = full * 127.5 + 127.5
    blur = blur * 127.5 + 127.5
    generated = generated * 127.5 + 127.5
    for i in range(generated.shape[0]):
        image_full = full[i, :, :, :]
        image_blur = blur[i, :, :, :]
        image_generated = generated[i, :, :, :]
        image = np.concatenate((image_full, image_blur, image_generated), axis=1)
        if (epoch is not None) and (index is not None):
            Image.fromarray(image.astype(np.uint8)).save(path + str(epoch + 1) + "_" + str(index + 1) + ".png")
        else:
            Image.fromarray(image.astype(np.uint8)).save(path + str(i) + ".png")


if __name__ == '__main__':
    format_image('data/small/test/301.jpg', size=256)
    build_hdf5('data/small')
    img_full, img_blur = load_data('train')
    print(img_full, '\n', len(img_blur))
