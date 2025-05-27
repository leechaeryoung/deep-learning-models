import os
import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, Add, BatchNormalization, LeakyReLU, Reshape, Flatten, Dense
from keras.optimizers import Adam
import numpy as np
from random import sample
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras import layers
import cv2
from google.colab.patches import cv2_imshow
import torch

print(f'keras : {keras.__version__}')
print(f'tensorflow : {tf.__version__}')
print(f'numpy : {np.__version__}')
print(f'cv2 : {cv2.__version__}')
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Version : {torch.version.cuda}")
print(f"cudnn Version : {torch.backends.cudnn.version()}")
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"Count of GPUs : {torch.cuda.device_count()}")


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


train_noisy_data = np.load('/content/drive/MyDrive/data/npy_best_friend/train_imgs.npy')
train_mask = np.load('/content/drive/MyDrive/data/npy_best_friend/train_mask.npy')
val_noisy_data = np.load('/content/drive/MyDrive/data/npy_best_friend/validation_imgs.npy')
val_mask = np.load('/content/drive/MyDrive/data/npy_best_friend/validation_mask.npy')
test_noisy_data = np.load('/content/drive/MyDrive/data/npy_best_friend/test_imgs.npy')
test_mask = np.load('/content/drive/MyDrive/data/npy_best_friend/test_mask.npy')

print(train_noisy_data.shape)
print(train_mask.shape)
print(val_noisy_data.shape)
print(val_mask.shape)
print(test_noisy_data.shape)
print(test_mask.shape)

train_noisy_img = train_noisy_data[0, :, :]
train_img = train_mask[0, :, :]
val_noisy_img = val_noisy_data[0, :, :]
val_img = val_mask[0, :, :]
test_noisy_img = test_noisy_data[0, :, :]
test_img = test_mask[0, :, :]

cv2_imshow(train_noisy_img)
cv2_imshow(train_img)
cv2_imshow(val_noisy_img)
cv2_imshow(val_img)
cv2_imshow(test_noisy_img)
cv2_imshow(test_img)

print(train_noisy_data.shape)
print(train_mask.shape)
print(val_noisy_data.shape)
print(val_mask.shape)
print(test_noisy_data.shape)
print(test_mask.shape)

train_noisy_data = train_noisy_data.astype("float32") / 255.0
train_mask = train_mask.astype("float32") / 255.0
val_noisy_data = val_noisy_data.astype("float32") / 255.0
val_mask = val_mask.astype("float32") / 255.0
test_noisy_data = test_noisy_data.astype("float32") / 255.0
test_mask = test_mask.astype("float32") / 255.0

print(train_noisy_data.shape)
print(train_mask.shape)
print(val_noisy_data.shape)
print(val_mask.shape)
print(test_noisy_data.shape)
print(test_mask.shape)

def display(array1, array2):
    """Displays ten random images from each array."""
    n = 10
    indices = np.random.randint(len(array1), size=n)
    print(indices)

    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(100, 20))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(256, 256))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(256, 256))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


input = layers.Input(shape=(256, 256, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
model_1 = Model(input, x)

model_1.summary()


model_1.compile(optimizer=Adam(0.001,beta_1=0.9), loss='binary_crossentropy',metrics=['accuracy'])
history_2 = model_1.fit(train_noisy_data, train_mask, batch_size = 32, epochs = 500,
                        validation_data=(val_noisy_data, val_mask), shuffle=True)

model_1.save('/content/drive/MyDrive/data/npy_best_friend/model_autoencoder.keras')


import time

# 추론 시간 측정
start_time = time.time()
imgs_test_pred = model_1.predict(test_noisy_data)
end_time = time.time()

# 이미지 디스플레이
display(test_noisy_data, imgs_test_pred)
np.save('/content/drive/MyDrive/data/npy_best_friend/imgs_test_pred_autoencoder.npy', imgs_test_pred)
print(test_noisy_data.shape)
print(imgs_test_pred.shape)

# 시간 계산 (ms 단위)
total_time = (end_time - start_time) * 1000  # 전체 추론 시간(ms)
average_time_per_inference = total_time / len(test_noisy_data)  # 영상 1장당 평균 시간(ms)

# 결과 출력
print(f"Total Inference Time: {total_time:.3f} ms")
print(f"Time per Inference Step (ms): {average_time_per_inference:.3f} ms")


print(train_noisy_data.shape)
print(train_mask.shape)
print(val_noisy_data.shape)
print(val_mask.shape)
print(test_noisy_data.shape)
print(test_mask.shape)
imgs_test_pred = np.squeeze(imgs_test_pred, axis=-1)
print(imgs_test_pred.shape)


size = test_noisy_data.shape[0] // 5
w = test_noisy_data.shape[2]
h = test_noisy_data.shape[1]
t_img = np.zeros((3 * h, 5 * w), np.uint8)
for i in range(5):
    t_img[0:h, i * w:i * w + w] = 255 * test_noisy_data[i,:,:]
    t_img[h:h + h, i * w:i * w + w] = 255 * test_mask[i,:,:]
    t_img[h + h:h + h + h, i * w:i * w + w] = 255 * imgs_test_pred[i,:,:]
# cv2.putText(t_img, 'noise o', (0, 20), 2, 1, (192, 192, 192))
# cv2.putText(t_img, 'noise x', (0, 256 + 20), 2, 1, (192, 192, 192))
# cv2.putText(t_img, 'result', (0, 512 + 20), 2, 1, (192, 192, 192))

cv2_imshow(t_img)


# plot the train and validation loss
plt.plot(history_2.history['loss'], 'b+')
plt.plot(history_2.history['val_loss'], 'g+')
plt.title('Train and Validation loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Validation loss'], loc='upper right')
plt.show()


from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_squared_error

psnr_value = peak_signal_noise_ratio(test_mask, imgs_test_pred)
ssim_loss = structural_similarity(test_mask, imgs_test_pred, channel_axis=-1, data_range=test_mask.max()-test_mask.min())
mse_value = mean_squared_error(test_mask.flatten(), imgs_test_pred.flatten())

print("**이미지 비교**")
print("PSNR :", psnr_value)
print("SSIM :", ssim_loss)
print("MSE :", mse_value)
