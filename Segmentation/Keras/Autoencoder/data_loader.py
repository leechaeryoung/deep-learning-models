import numpy as np

def load_data():
    path = '/content/drive/MyDrive/npy_friend/'

    train_mask = np.load(path + 'train_mask.npy').astype("float32") / 255.0
    train_imgs = np.load(path + 'train_imgs.npy').astype("float32") / 255.0
    test_imgs = np.load(path + 'test_imgs.npy').astype("float32") / 255.0
    test_mask = np.load(path + 'test_mask.npy').astype("float32") / 255.0

    return train_imgs, train_mask, test_imgs, test_mask
