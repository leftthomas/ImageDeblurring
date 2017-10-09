import os
from pathlib import Path

import h5py
import numpy as np
import parmap
from PIL import Image
from tqdm import tqdm


def normalization(x):
    return x / 127.5 - 1


def format_image(img_path, size):
    """
    Load image with PIL and reshape
    """

    img = Image.open(img_path)
    # slice image in 2 to get both parts
    img_full = img.crop((0, 0, img.size[0] / 2, img.size[1]))
    img_blur = img.crop((img.size[0] / 2, 0, img.size[0], img.size[1]))

    # img_full.show()
    # img_blur.show()

    img_full = img_full.resize((size, size), Image.ANTIALIAS)
    img_blur = img_blur.resize((size, size), Image.ANTIALIAS)

    img_full = np.expand_dims(img_full, 0)
    img_blur = np.expand_dims(img_blur, 0)

    # print(img_full.shape)
    # print(img_blur.shape)

    return img_full, img_blur


# format_image('data/test/301.jpg',size=256)


def build_hdf5(jpeg_dir, size=256):
    """
    Gather the data in a single HDF5 file.
    """
    # put train data in HDF5
    hdf5_file = os.path.join('data', "data.h5")
    with h5py.File(hdf5_file, "w") as hfw:

        for data_type in ["train", "test", "val"]:

            list_img = list(Path(jpeg_dir).glob('%s/*.jpg' % data_type))
            list_img.extend(list(Path(jpeg_dir).glob('%s/*.png' % data_type)))
            list_img = map(str, list_img)
            list_img = np.array(list_img)

            data_full = hfw.create_dataset("%s_data_full" % data_type, (0, 3, size, size),
                                           maxshape=(None, 3, size, size), dtype=np.uint8)

            data_sketch = hfw.create_dataset("%s_data_sketch" % data_type, (0, 3, size, size),
                                             maxshape=(None, 3, size, size), dtype=np.uint8)

            num_files = len(list_img)
            chunk_size = 100
            num_chunks = num_files / chunk_size
            arr_chunks = np.array_split(np.arange(num_files), num_chunks)

            for chunk_idx in tqdm(arr_chunks):
                list_img_path = list_img[chunk_idx].tolist()
                output = parmap.map(format_image, list_img_path, size, 3, parallel=False)

                arr_img_full = np.concatenate([o[0] for o in output], axis=0)
                arr_img_sketch = np.concatenate([o[1] for o in output], axis=0)

                # Resize HDF5 dataset
                data_full.resize(data_full.shape[0] + arr_img_full.shape[0], axis=0)
                data_sketch.resize(data_sketch.shape[0] + arr_img_sketch.shape[0], axis=0)

                data_full[-arr_img_full.shape[0]:] = arr_img_full.astype(np.uint8)
                data_sketch[-arr_img_sketch.shape[0]:] = arr_img_sketch.astype(np.uint8)


# def check_HDF5(jpeg_dir, nb_channels):
#     """
#     Plot images with landmarks to check the processing
#     """
#
#     # Get hdf5 file
#     file_name = os.path.basename(jpeg_dir.rstrip("/"))
#     hdf5_file = os.path.join(data_dir, "%s_data.h5" % file_name)
#
#     with h5py.File(hdf5_file, "r") as hf:
#         data_full = hf["train_data_full"]
#         data_sketch = hf["train_data_sketch"]
#         for i in range(data_full.shape[0]):
#             plt.figure()
#             img = data_full[i, :, :, :].transpose(1,2,0)
#             img2 = data_sketch[i, :, :, :].transpose(1,2,0)
#             img = np.concatenate((img, img2), axis=1)
#             if nb_channels == 1:
#                 plt.imshow(img[:, :, 0], cmap="gray")
#             else:
#                 plt.imshow(img)
#             plt.show()
#             plt.clf()
#             plt.close()
#

def load_data():
    with h5py.File("data/data.h5", "r") as hf:
        X_full_train = hf["train_data_full"][:].astype(np.float32)
        X_full_train = normalization(X_full_train)

        X_sketch_train = hf["train_data_sketch"][:].astype(np.float32)
        X_sketch_train = normalization(X_sketch_train)

        X_full_train = X_full_train.transpose(0, 2, 3, 1)
        X_sketch_train = X_sketch_train.transpose(0, 2, 3, 1)

        X_full_val = hf["val_data_full"][:].astype(np.float32)
        X_full_val = normalization(X_full_val)

        X_sketch_val = hf["val_data_sketch"][:].astype(np.float32)
        X_sketch_val = normalization(X_sketch_val)

        X_full_val = X_full_val.transpose(0, 2, 3, 1)
        X_sketch_val = X_sketch_val.transpose(0, 2, 3, 1)

        return X_full_train, X_sketch_train, X_full_val, X_sketch_val
