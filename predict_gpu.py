from PIL import Image

from datasets import *
import torch, cv2
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
import os

from model import YUV_Net


loader = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
unloader = transforms.ToPILImage()

def image_loader(image_name):
    # image = Image.open(image_name).resize((512,512),Image.ANTIALIAS).convert('RGB')
    img = cv2.imread(image_name)
    h, w, _ = img.shape
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    img = Image.fromarray(img)
    # image = Image.open(image_name)
    image = loader(img).unsqueeze(0)
    return image.to(torch.float), (w, h)

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    # step1：首先通过squeez()将输入的tensor去掉batch这个维度,变为[c,h,w],
    # 之后转为float和cpu，再将tensor的值限制在[0,1]之间（有时候tensor为负值）
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    # step1：将tensor进行一个线性拉伸，拉伸到最大值为1，最小值为0
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:  # 一般情况下会直接跳到这里
        img_np = tensor.detach().numpy()  # 转为array形式
        img_np = np.transpose(
            img_np[[2, 1, 0], :, :], (1, 2, 0)
        )  # CHW->HWC, RGB->BGR（因为后续要用cv2保存）
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()  # 乘以255转为uint8
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def tensor_to_np(x, min_max=(0, 1)):
    # tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    # img = tensor.mul(255).byte()
    # img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))

    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    x[0] = x[0] * std[0] + mean[0]
    x[1] = x[1] * std[1] + mean[1]
    x[2] = x[2].mul(std[2]) + mean[2]

    img = x.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img


gen = U_Net().eval().cuda()

# gen=GCANet(in_c=3)
# gen.load_state_dict(torch.load('/home/admin1/SUEP_ZTY/Mine_GAN_1/saved_models/haze/generator_1.pth',map_location=torch.device('cpu')))
gen.load_state_dict(
    torch.load(
        "/home/admin1/SUEP_ZTY/Mine_GAN_1/saved_models/haze_OTS+/generator_9.pth"
    )
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(img_path):
    img, img_shape = image_loader(img_path)
    img = img.to(device)
    new_img = gen(img)
    new_img = new_img.squeeze(0)
    new_img = tensor_to_np(new_img)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_YUV2RGB)
    new_img = cv2.resize(new_img, img_shape)
    return new_img
    # original_img = cv2.imread(img_path)
    # cv2.imshow('original',original_img)
    # cv2.imshow('output',new_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


# loader_=transforms.ToTensor()

img_path = "./hazy_/"
save_path = "./predict_out/"
i = 0
for name in os.listdir(img_path):
    i += 1
    read = img_path + name
    save = save_path + name
    print(i, name)
    dehaze_img = predict(read)
    cv2.imwrite(save, dehaze_img)

# predict(img_path)
