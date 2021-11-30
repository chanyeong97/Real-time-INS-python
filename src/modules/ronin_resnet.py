import math
import tensorflow as tf

from glob import glob
from os import path as osp

from src.modules.transformations import RandomHoriRotate
from src.modules.data_load import GlobSpeedSequence, RoninResnetDataset
from src.models.resnet import ResNet18


def get_dataset(data_list, args, **kwargs):
    random_shift, shuffle, transform, grv_only = 0, False, None, False
    if args.mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transform = RandomHoriRotate(math.pi * 2)
    elif args.mode == 'val':
        shuffle = True
    elif args.mode == 'test':
        shuffle = False
        grv_only = True

    seq_type = GlobSpeedSequence
    dataset = RoninResnetDataset(
        seq_type=seq_type, step_size=args.step_size, window_size=args.window_size, 
        random_shift=random_shift, transform=transform, data_list=data_list, 
        shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error)
    return dataset


def get_dataset_from_list(dataset, args, **kwargs):
    data_list = glob(osp.join(dataset, '*'))
    return get_dataset(data_list, args, **kwargs)

def train_ronin_resnet(args):
    train_dataset = get_dataset_from_list(args.train_dataset, args, mode='train')
    
    if not args.validation_dataset == None:
        validation_dataset = get_dataset_from_list(args.validation_dataset, args, mode='val')
    
    start_epoch = 0
    model = ResNet18()
    loss = tf.losses.MSE()
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, args.model_path, max_to_keep=10)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        start_epoch = manager.latest_checkpoint
    else:
        print("Initializing from scratch.")

    for epoch in range(start_epoch, args.epochs):
        loss = []


def test_ronin_resnet(args):
    test_dataset = get_dataset_from_list(args.test_dataset, args, mode='test')

