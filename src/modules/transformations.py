import numpy as np
import math


class RandomHoriRotate:
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, feat, targ, **kwargs):
        angle = np.random.random() * self.max_angle
        rm = np.array([[math.cos(angle), -math.sin(angle)],
                       [math.sin(angle), math.cos(angle)]])
        feat_aug = np.copy(feat)
        targ_aug = np.copy(targ)
        feat_aug[:, :2] = np.matmul(rm, feat[:, :2].T).T
        feat_aug[:, 3:5] = np.matmul(rm, feat[:, 3:5].T).T
        targ_aug[:2] = np.matmul(rm, targ[:2].T).T

        return feat_aug, targ_aug