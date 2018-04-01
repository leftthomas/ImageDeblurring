# ImageDeblurring
A Keras implementation of image deblurring based on ICCV 2017 paper 
[Deep Generative Filter for motion deblurring](https://arxiv.org/abs/1709.03481)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- TensorFlow
```
conda create -n tensorflow
source activate tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp36-cp36m-linux_x86_64.whl
```
- Keras
```
pip install keras
```
- tqdm
```
conda install tqdm
```

## Usage

### Train

```
python main.py
```
