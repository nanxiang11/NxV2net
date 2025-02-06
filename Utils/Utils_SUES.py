import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.train_image_list = sorted([f for f in os.listdir(os.path.join(data_dir, 'images'))])
        self.label_image_list = sorted([f for f in os.listdir(os.path.join(data_dir, 'masks'))])

        # 检查图像和标签文件名称是否一致
        self._check_filenames()

    def __len__(self):
        return len(self.train_image_list)

    def __getitem__(self, idx):
        train_image_name = self.train_image_list[idx]
        train_image_path = os.path.join(self.data_dir, 'images', train_image_name)  # 训练图像路径
        train_image = Image.open(train_image_path)
        train_image = ImageEnhance.Brightness(train_image).enhance(1.2)
        train_image = ImageEnhance.Contrast(train_image).enhance(1.2)

        label_image_name = self.label_image_list[idx]
        label_image_path = os.path.join(self.data_dir, 'masks', label_image_name)  # 目标图像路径
        label_image = Image.open(label_image_path).convert('L')

        if self.transform:
            train_image = self.transform(train_image)
            label_image = self.transform(label_image)

        return (train_image, label_image)

    def _check_filenames(self):
        # 检查图像和标签文件名称是否一致
        if self.train_image_list != self.label_image_list:
            raise ValueError("The filenames in 'images' and 'masks' directories do not match or are not in the same order.")

def get_transforms():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
    ])
    return transform



class DrawImage:

    def test_loaderImage(self, dataset, num=6):
        fig, axs = plt.subplots(2, num, figsize=(10, 6))
        indices = random.sample(range(len(dataset)), num)  # 随机抽取num个索引
        for i, idx in enumerate(indices):
            train_image, label_image = dataset[idx]
            train_image, label_image = np.transpose(train_image.numpy(), (1, 2, 0)), np.transpose(label_image.numpy(), (1, 2, 0))
            axs[0, i].imshow(train_image.squeeze(), cmap='gray')
            axs[0, i].set_title("image")  # 设置标题
            axs[0, i].axis('off')  # 关闭坐标轴

            axs[1, i].imshow(label_image.squeeze(), cmap='gray')
            axs[1, i].set_title("mask")  # 设置标题
            axs[1, i].axis('off')  # 关闭坐标轴
        plt.show()

    def predict_Image(testset, net, num=6):
        fig, axs = plt.subplots(3, num, figsize=(10, 6))
        for i in range(num):
            train_image, label_image = testset[i]
            pre_image = net(train_image.unsqueeze(0))
            train_image, label_image, pre_image = (np.transpose(train_image.cpu().detach().numpy(), (1, 2, 0)),
                                       np.transpose(label_image.cpu().detach().numpy(), (1, 2, 0)),
                                       np.transpose(pre_image.cpu().detach().numpy()))

            axs[0, i].imshow(train_image.squeeze(), cmap='gray')
            axs[0, i].set_title("image")  # 设置标题
            axs[0, i].axis('off')  # 关闭坐标轴

            axs[1, i].imshow(label_image.squeeze(), cmap='gray')
            axs[1, i].set_title("mask")  # 设置标题
            axs[1, i].axis('off')  # 关闭坐标轴

            axs[2, i].imshow(label_image.squeeze(), cmap='gray')
            axs[2, i].set_title("pre")  # 设置标题
            axs[2, i].axis('off')  # 关闭坐标轴

        plt.show()



# dataset = CustomDataset("D:/CV/Data/crack_segmentation_dataset/crack_segmentation_dataset/", transform=get_transforms())
# print(dataset.__len__())
# drawImage = DrawImage()
# drawImage.test_loaderImage(dataset)

