import numpy as np

def load_data():
    data_path = '/content/drive/MyDrive/npy_friend/'
    train_noisy = np.load(data_path + 'train_imgs.npy').astype("float32") / 255.0
    train_mask = np.load(data_path + 'train_mask.npy').astype("float32") / 255.0
    test_noisy = np.load(data_path + 'test_imgs.npy').astype("float32") / 255.0
    test_mask = np.load(data_path + 'test_mask.npy').astype("float32") / 255.0
    return train_noisy, train_mask, test_noisy, test_mask
