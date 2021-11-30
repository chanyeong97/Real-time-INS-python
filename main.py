import argparse

from src.modules.ronin_resnet import train_ronin_resnet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, choices=['ronin', 'oxiod'], default='ronin')
    parser.add_argument('--train_dataset', type=str, default='dataset/RoNIN_dataset/train_dataset')
    parser.add_argument('--validation_dataset', type=str, default='dataset/RoNIN_dataset/seen_subjects_test_set')
    parser.add_argument('--test_dataset', type=str, default='dataset/RoNIN_dataset/unseen_subjects_test_set')
    parser.add_argument('--model_path', type=str, default='pretrained_models/my_resnet/model/model')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--arch', type=str, choices=['ronin_resnet', 'my_resnet', 'my_cnn'], default='ronin_resnet')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-04)
    return parser.parse_args()


def train(args):
    if args.arch == 'ronin_resnet':
        train_ronin_resnet(args)


def test(args):
    print('test')


if __name__ == '__main__':
    args = parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)