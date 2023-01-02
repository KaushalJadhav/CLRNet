import os
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
import argparse
import numpy as np
import random
from clrnet.utils.config import Config
from clrnet.engine.runner import CLRNet
from clrnet.datasets import build_dataloader


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed

    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs
    
    cfg.wandb_name = args.wandb_name
    cfg.wandb_project = args.wandb_project
    cfg.wandb_id = args.wandb_id
    
    cudnn.benchmark = True
    model=CLRNet()
    trainer=pl.Trainer()

    if args.validate:
        trainer.validate(model)
    elif args.test:
        test_dataloader=CLRNet.test_dataloader
        trainer.test(model,dataloaders=test_dataloader)
    else:

        trainer.fit(model)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dirs',
                        type=str,
                        default=None,
                        help='work dirs')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--resume_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--finetune_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--view', action='store_true', help='whether to view')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test',
        action='store_true',
        help='whether to test the checkpoint on testing set')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--wandb_name', type=str, default=None, help='Name of wandb run')
    parser.add_argument('--wandb_project', type=str, default=None, help='Name of wandb project')
    parser.add_argument('--wandb_id', type=str, default=None, help='Name of wandb id')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
