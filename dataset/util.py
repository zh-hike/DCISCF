from torchvision.transforms import functional as tf
from torchvision.transforms import transforms


class Crop(object):
    """
    图像裁剪处理，从图像左上角开始，裁剪256x256大小的
    """
    def __init__(self):
        self.w = 256
        self.h = 256

    def __call__(self, img):
        return tf.crop(img, 0, 0, self.w, self.h)


class TrainTransform:
    """
    训练集的图像transform
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Grayscale(1),      #转化为灰度图像
            Crop(),           # 裁剪
            transforms.ToTensor(),      # 转化为tensor
        ])

    def __call__(self, img):
        img = self.transform(img)
        return img


class EvalTransform:
    def __init__(self, time):
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            # transforms.RandomCrop(size=(256, 256), padding=0),
            Crop(),
            transforms.ToTensor(),
        ])
        self.time = time

    def __call__(self, img):
        imgs = []
        for i in range(self.time):
            imgs.append(self.transform(img))

        return imgs
