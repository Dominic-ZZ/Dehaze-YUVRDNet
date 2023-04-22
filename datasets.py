import glob
import random
import os
import numpy as np
import cv2
import torch
from PIL import Image

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.ToTensor()


def tensor2uv(tensor):
    y, u, v = torch.split(tensor, 1, dim=0)
    uv = torch.cat((u, v), dim=0)
    return uv


def tensor2y(tensor):
    y, u, v = torch.split(tensor, 1, dim=0)
    return y


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=transform, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):
        # img = Image.open(self.files[index % len(self.files)])
        img = cv2.imread(self.files[index % len(self.files)])
        yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

        # sobel=rgb2sobel(img)
        # y,u,v=cv2.split(yuv)
        # uv_img=cv2.merge((u,v))
        # suv_img=cv2.merge((sobel,u,v))

        img = Image.fromarray(yuv)

        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        # if np.random.random() < 0.5:
        #     img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
        #     img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        # img_A=tensor2uv(img_A)
        # img_B=tensor2uv(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)
