import numpy as np
import h5py
import json
from abc import ABC, abstractmethod

from os import path as osp

from src.modules.math_utils import gyro_integration


class CompiledSequence(ABC):
    """
    An abstract interface for compiled sequence.
    """
    def __init__(self, **kwargs):
        super(CompiledSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def get_aux(self):
        pass

    def get_meta(self):
        return "No info available"


def load_cached_sequences(seq_type, data_list, **kwargs):
    features_all, targets_all, aux_all = [], [], []
    for i in range(len(data_list)):
        seq = seq_type(data_list[i], **kwargs)
        feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
        print(seq.get_meta())
        features_all.append(feat)
        targets_all.append(targ)
        aux_all.append(aux)
    return features_all, targets_all, aux_all
        

def select_orientation_source(data_path, max_ori_error=20.0, grv_only=True, use_ekf=True):
    ori_names = ['gyro_integration', 'game_rv']
    ori_sources = [None, None, None]

    with open(osp.join(data_path, 'info.json')) as f:
        info = json.load(f)
        ori_errors = np.array(
            [info['gyro_integration_error'], info['grv_ori_error'], info['ekf_ori_error']])
        init_gyro_bias = np.array(info['imu_init_gyro_bias'])

    with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
        ori_sources[1] = np.copy(f['synced/game_rv'])
        if grv_only or ori_errors[1] < max_ori_error:
            min_id = 1
        else:
            if use_ekf:
                ori_names.append('ekf')
                ori_sources[2] = np.copy(f['pose/ekf_ori'])
            min_id = np.argmin(ori_errors[:len(ori_names)])
            if min_id == 0:
                ts = f['synced/time']
                gyro = f['synced/gyro_uncalib'] - init_gyro_bias
                ori_sources[0] = gyro_integration(ts, gyro, ori_sources[1][0])

    return ori_names[min_id], ori_sources[min_id], ori_errors[min_id]