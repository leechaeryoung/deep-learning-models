import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class SobelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out

# SobelConv2d 인스턴스 생성
in_channels = 3
out_channels = 4
sobel_conv = SobelConv2d(in_channels=in_channels, out_channels=out_channels)

# 이미지 데이터 준비
image_path = '/content/drive/MyDrive/SRI_data/thyroid/image_directory/2024-06-17_15-30-52_360-10.sri_off.bmp'
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)

# Sobel 필터 적용
with torch.no_grad():
    if torch.cuda.is_available():
        sobel_conv.cuda()
        image_tensor = image_tensor.cuda()

    sobel_output = sobel_conv(image_tensor)

print(sobel_output.shape)

# 출력 이미지 시각화
num_channels = sobel_output.shape[1]
fig, axs = plt.subplots(1, num_channels, figsize=(15, 5))

for i in range(num_channels):
    ax = axs[i]
    output_image = sobel_output[0, i].cpu().numpy()
    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    ax.imshow(output_image, cmap='gray')
    ax.axis('off')
    titles = ['horizontal', 'vertical', 'diagonal1', 'diagonal2']
    ax.set_title(titles[i])

plt.show()
