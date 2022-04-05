import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvBNRelu(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1):
        super(ConvBNRelu, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, img):
        img = self.net(img)

        return img


class BT2(nn.Module):
    def __init__(self):
        super(BT2, self).__init__()
        self.net = nn.Sequential(
            ConvBNRelu(16, 16),
            ConvBNRelu(16, 16, kernel_size=1, padding=0),
        )

    def forward(self, img):

        img = img + self.net(img)

        return img


class BT3(nn.Module):
    def __init__(self):
        super(BT3, self).__init__()
        self.block1 = ConvBNRelu(16, 16)
        self.block2 = ConvBNRelu(16, 16)
        self.block3 = ConvBNRelu(32, 16)

    def forward(self, img):
        im1 = self.block1(img)
        im1img = im1 + img
        im2 = self.block2(im1img)
        im2im1 = im2 + im1
        img3 = torch.cat([im2, im2im1], dim=1)
        img = img + self.block3(img3)
        return img


class BT4(nn.Module):
    def __init__(self, in_features, out_features):
        super(BT4, self).__init__()
        self.net = nn.Sequential(
            ConvBNRelu(in_features, out_features),
            ConvBNRelu(out_features, out_features),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.net2 = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, img):

        img = self.net(img) + self.net2(img)
        return img


class BT5(nn.Module):
    def __init__(self):
        super(BT5, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.net = nn.Sequential(

            nn.Linear(128, 3),
            # nn.Softmax(dim=1),
        )

    def forward(self, img):
        n, _, _, _ = img.shape
        img = self.pool(img).view(n, -1)
        img = self.net(img)
        return img


class Block1(nn.Module):
    def __init__(self):
        super(Block1, self).__init__()

        self.net = nn.Sequential(
            ConvBNRelu(1, 16),
            ConvBNRelu(16, 16),
        )

    def forward(self, img):
        return self.net(img)


class Block2(nn.Module):
    def __init__(self):
        super(Block2, self).__init__()
        self.net = nn.Sequential(
            BT2(),
            BT2(),
            BT2(),
        )

    def forward(self, img):
        img = self.net(img)

        return img


class Block3(nn.Module):
    def __init__(self):
        super(Block3, self).__init__()
        self.net = nn.Sequential(
            BT3(),
            BT3(),
        )

    def forward(self, img):
        img = self.net(img)
        return img


class Block4(nn.Module):
    def __init__(self):
        super(Block4, self).__init__()
        self.net = nn.Sequential(
            BT4(16, 32),
            BT4(32, 64),
            BT4(64, 128),
        )

    def forward(self, img):
        img = self.net(img)
        return img


class DCISCF(nn.Module):
    def __init__(self):
        super(DCISCF, self).__init__()
        self.net = nn.Sequential(
            Block1(),
            Block2(),
            Block3(),
            Block4(),
            BT5(),
        )

    def forward(self, img):
        output = self.net(img)

        return output
