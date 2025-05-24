import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, Add, BatchNormalization, LeakyReLU, Reshape, Flatten, Dense
from keras.optimizers import Adam
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras import layers
from random import sample
from tensorflow.python.client import device_lib

# 버전 확인
print(f'keras : {keras.__version__}')
print(f'tensorflow : {tf.__version__}')
print(f'numpy : {np.__version__}')
print(f'cv2 : {cv2.__version__}')
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Version : {torch.version.cuda}")
print(f"cuDNN Version : {torch.backends.cudnn.version()}")
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"Count of GPUs : {torch.cuda.device_count()}")
print(device_lib.list_local_devices())

# 설정
img_rows = 256
img_cols = 256
ch = 1
data_path = '/content/drive/MyDrive/npy_friend/'

# 데이터 로드
def load_train_data(data_path):
    imgs_train = np.load(data_path + 'train_imgs.npy')
    imgs_mask_train = np.load(data_path + 'train_mask.npy')
    return imgs_train, imgs_mask_train

def load_test_data(data_path):
    imgs_test = np.load(data_path + 'test_imgs.npy')
    imgs_mask_test = np.load(data_path + 'test_mask.npy')
    return imgs_test, imgs_mask_test

# 이미지 전처리
def preprocess(imgs, img_rows, img_cols):
    imgs_p = np.ndarray((imgs.shape[0], 1, img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        tmp1 = cv2.resize(imgs[i], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 0, :, :] = tmp1
    return imgs_p

def to_tensor(imgs):
    imgs = imgs.astype('float32') / 255.0
    return torch.tensor(imgs, dtype=torch.float32)

# Dataset 클래스
class UltrasoundDataset(Dataset):
    def __init__(self, noisy_imgs, clean_imgs):
        self.noisy_imgs = noisy_imgs
        self.clean_imgs = clean_imgs

    def __len__(self):
        return len(self.noisy_imgs)

    def __getitem__(self, idx):
        return self.noisy_imgs[idx], self.clean_imgs[idx]

# 데이터 로딩
imgs_train, imgs_mask_train = load_train_data(data_path)
imgs_test, imgs_mask_test = load_test_data(data_path)
print(f"Shape of imgs_train: {imgs_train.shape}")
print(f"Shape of imgs_mask_train: {imgs_mask_train.shape}")
print(f"Shape of imgs_test: {imgs_test.shape}")
print(f"Shape of imgs_mask_test: {imgs_mask_test.shape}")
print("----------------------------------------------------------------------")

# 전처리
imgs_train_p = preprocess(imgs_train, img_rows, img_cols)
imgs_mask_train_p = preprocess(imgs_mask_train, img_rows, img_cols)
imgs_test_p = preprocess(imgs_test, img_rows, img_cols)
imgs_mask_test_p = preprocess(imgs_mask_test, img_rows, img_cols)
print(f"Shape of imgs_train_p: {imgs_train_p.shape}")
print(f"Shape of imgs_mask_train_p: {imgs_mask_train_p.shape}")
print(f"Shape of imgs_test_p: {imgs_test_p.shape}")
print(f"Shape of imgs_mask_test_p: {imgs_mask_test_p.shape}")
print("----------------------------------------------------------------------")

# train/validation 분리
imgs_train_p, val_imgs, imgs_mask_train_p, val_masks = train_test_split(
    imgs_train_p, imgs_mask_train_p, test_size=0.2, random_state=42)
print(f"Shape of imgs_train_p: {imgs_train_p.shape}")
print(f"Shape of imgs_mask_train_p: {imgs_mask_train_p.shape}")
print(f"Shape of val_imgs: {val_imgs.shape}")
print(f"Shape of val_masks: {val_masks.shape}")
print("----------------------------------------------------------------------")

# Tensor 변환
imgs_train_tensor = to_tensor(imgs_train_p)
imgs_mask_train_tensor = to_tensor(imgs_mask_train_p)
val_imgs_tensor = to_tensor(val_imgs)
val_masks_tensor = to_tensor(val_masks)
imgs_test_tensor = to_tensor(imgs_test_p)
imgs_mask_test_tensor = to_tensor(imgs_mask_test_p)
print(f"Shape of imgs_train_tensor: {imgs_train_tensor.shape}")
print(f"Shape of imgs_mask_train_tensor: {imgs_mask_train_tensor.shape}")
print(f"Shape of val_imgs_tensor: {val_imgs_tensor.shape}")
print(f"Shape of val_masks_tensor: {val_masks_tensor.shape}")
print(f"Shape of imgs_test_tensor: {imgs_test_tensor.shape}")
print(f"Shape of imgs_mask_test_tensor: {imgs_mask_test_tensor.shape}")
print("----------------------------------------------------------------------")

# DataLoader
train_dataset = UltrasoundDataset(imgs_train_tensor, imgs_mask_train_tensor)
val_dataset = UltrasoundDataset(val_imgs_tensor, val_masks_tensor)
test_dataset = UltrasoundDataset(imgs_test_tensor, imgs_mask_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
