
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Load functions
def load_train_data(data_path):
    imgs_train = np.load(data_path + 'train_imgs.npy')
    imgs_mask_train = np.load(data_path + 'train_mask.npy')
    return imgs_train, imgs_mask_train

def load_test_data(data_path):
    imgs_test = np.load(data_path + 'test_imgs.npy')
    imgs_mask_test = np.load(data_path + 'test_mask.npy')
    return imgs_test, imgs_mask_test

def preprocess(imgs, img_rows, img_cols):
    imgs_p = np.ndarray((imgs.shape[0], 1, img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        tmp1 = cv2.resize(imgs[i], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 0, :, :] = tmp1
    return imgs_p

def to_tensor(imgs):
    imgs = imgs.astype('float32') / 255.0
    return torch.tensor(imgs, dtype=torch.float32)

class UltrasoundDataset(Dataset):
    def __init__(self, noisy_imgs, clean_imgs):
        self.noisy_imgs = noisy_imgs
        self.clean_imgs = clean_imgs

    def __len__(self):
        return len(self.noisy_imgs)

    def __getitem__(self, idx):
        return self.noisy_imgs[idx], self.clean_imgs[idx]
