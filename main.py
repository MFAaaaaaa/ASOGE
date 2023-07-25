import argparse
import os
from train import Trainer
import torch


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='3', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_class', default=31, type=int)
    parser.add_argument('--source_model_path', default='./model_source/20220425-2142-webcam9_1_resnet50_best.pkl', type=str,
                        help='path to the pre-trained source model')
    # 20220425-2141-dslr9_1_resnet50_best.pkl
    # 20220425-2138-amazon9_1_resnet50_best.pkl
    # 20220425-2142-webcam9_1_resnet50_best.pkl
    parser.add_argument('--max_epoch', default=100, type=int)
    # parser.add_argument('--generator_epoch', default=1000, type=int)
    parser.add_argument('--data_path', default='./dataset/Office-31/amazon/images', type=str,
                        help='path to target data')
    parser.add_argument('--label_path', default='./data_utils/amazon.pkl', type=str)

    args = parser.parse_args()
    return args


torch.multiprocessing.set_sharing_strategy('file_system')
args = arg_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
oct_trainer = Trainer(args)
oct_trainer.train()
