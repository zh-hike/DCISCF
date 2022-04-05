from torchvision.transforms import functional as tf
from torchvision.transforms import transforms


class Crop(object):
    def __init__(self):
        self.w = 256
        self.h = 256

    def __call__(self, img):
        return tf.crop(img, 0, 0, self.w, self.h)


class TrainTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            Crop(),
            transforms.ToTensor(),
        ])

    def __call__(self, img):
        img = self.transform(img)
        return img


class EvalTransform:
    def __init__(self, time):
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.RandomCrop(size=(256, 256), padding=0),
            # Crop(),
            transforms.ToTensor(),
        ])
        self.time = time

    def __call__(self, img):
        imgs = []
        for i in range(self.time):
            imgs.append(self.transform(img))

        return imgs
