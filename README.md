# ImageDeblurring
A Keras implementation of image deblurring based on ICCV 2017 paper 
[Deep Generative Filter for motion deblurring](https://arxiv.org/pdf/1709.03481.pdf)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- TensorFlow
```
conda create -n tensorflow
source activate tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0rc0-cp34-cp34m-linux_x86_64.whl
```
- Keras
```
pip install keras
```
- tqdm
```
pip install tqdm
```
- opencv
```
conda install opencv
```

## Usage

### Train

```
python train.py
```
