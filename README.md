# Real-time-INS-python

Tensorflow를 이용한 딥러닝 기반 관성항법시스템입니다

## Requirements

- python3
- numpy
- scipy
- numba
- numpy-quaternion
- tensorflow>=2.2

## 데이터

- [RoNIN](https://github.com/Sachini/ronin/blob/master/README.md) dataset
- OXIOD dataset

## 학습

- RoNIN ResNet-18
```bash
python main.py --mode train --arch ronin_resnet --train_dataset <RoNIN train dataset 경로> --validation_dataset <RoNIN validation dataset 경로>
```


## 평가

- RoNIN ResNet-18
```bash
python main.py --mode test --arch ronin_resnet --test_dataset <RoNIN test dataset 경로>
```
