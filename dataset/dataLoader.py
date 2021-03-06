import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from dataset.util import Crop
from dataset.util import TrainTransform, EvalTransform


class DL:
    """
    加载数据集，训练集和验证集
    同时设置dataloader

    """
    def __init__(self, args):
        train_transform = TrainTransform()
        eval_transform = EvalTransform(args.time)
        train_data = ImageFolder(args.data_path + 'train/', transform=train_transform)
        eval_data = ImageFolder(args.data_path + 'eval/', transform=eval_transform)

        test_data = ImageFolder(args.data_path + 'test/', transform=eval_transform)


        self.traindl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        self.evaldl = DataLoader(eval_data, batch_size=20)
        self.testdl = DataLoader(test_data, batch_size=20)
        self.test_name = test_data.imgs
        self.test_classes = train_data.classes
