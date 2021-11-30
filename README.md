# Real-time-INS-python

Tensorflow를 이용한 실시간 딥러닝 기반 관성항법시스템입니다.

## Requirements

- python3
- numpy
- scipy
- numba
- numpy-quaternion
- tensorflow>=2.2

## Data

- [RoNIN](https://github.com/Sachini/ronin/blob/master/README.md) dataset
- OXIOD dataset

## Train

- RoNIN ResNet-18```python main.py --mode train --arch ronin_resnet --train_dataset <RoNIN train dataset 경로> --validation_dataset <RoNIN validation dataset 경로>```


## Test

- RoNIN ResNet-18```python main.py --mode test --arch ronin_resnet --test_dataset <RoNIN test dataset 경로>```
