import math
import tensorflow as tf
import numpy as np

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
        random_shift=random_shift, transform=transform, data_list=data_list, batch_size=args.batch_size, 
        shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error)
    return dataset


def get_dataset_from_list(dataset, args, **kwargs):
    data_list = glob(osp.join(dataset, '*'))
    return get_dataset(data_list, args, **kwargs)

def train_ronin_resnet(args):
    train_dataset = get_dataset_from_list(args.train_dataset, args, mode='train')
    
    if not args.validation_dataset == None:
        validation_dataset = get_dataset_from_list(args.validation_dataset, args, mode='val')

    best_val_loss = np.inf
    start_epoch = 0
    model = ResNet18()
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, args.model_path, max_to_keep=10)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        validation_dataset.random_shuffle()
        val_total_loss = []
        for i in range(100):#len(validation_dataset)//args.batch_size):
            feat, targ, _, _ = validation_dataset(i)
            val_pred = model(feat, training=False)
            val_loss = tf.reduce_mean(tf.keras.losses.mse(val_pred, targ))
            val_total_loss.append(val_loss.numpy())
        best_val_loss = sum(val_total_loss) / len(val_total_loss)
        start_epoch = int(ckpt.step)
    else:
        print("Initializing from scratch.")

    for epoch in range(start_epoch, args.epochs):
        total_loss = []
        train_dataset.random_shuffle()
        for i in range(100):#len(train_dataset) // args.batch_size):
            feat, targ, _, _ = train_dataset(i)
            with tf.GradientTape() as g:
                pred = model(feat, training=True)
                loss = tf.reduce_mean(tf.keras.losses.mse(pred, targ))
            
            trainable_variables = model.trainable_variables
            gradients = g.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
            total_loss.append(loss.numpy())
            print(i,'/',len(train_dataset)//args.batch_size)
        avg_loss = sum(total_loss) / len(total_loss)
        ckpt.step.assign_add(1)

        if not args.validation_dataset == None:
            validation_dataset.random_shuffle()
            val_total_loss = []
            for i in range(100):#len(validation_dataset)//args.batch_size):
                feat, targ, _, _ = validation_dataset(i)
                val_pred = model(feat, training=False)
                val_loss = tf.reduce_mean(tf.keras.losses.mse(val_pred, targ))
                val_total_loss.append(val_loss.numpy())
            val_avg_loss = sum(val_total_loss) / len(val_total_loss)
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                model.save_weights(args.model_path)


def test_ronin_resnet(args):
    test_dataset = get_dataset_from_list(args.test_dataset, args, mode='test')
    model = ResNet18()
    model.load_weights(args.model_path)
    test_total_loss = []
    for i in range(100):#len(test_dataset)//args.batch_size:
        feat, targ, _, _ = test_dataset(i)
        test_pred = model(feat, training=False)
        test_loss = tf.reduce_mean(tf.keras.losses.mse(test_pred, targ))
        test_total_loss.append(test_loss.numpy())
    test_avg_loss = sum(test_total_loss) / len(test_total_loss)
    print(test_avg_loss)
