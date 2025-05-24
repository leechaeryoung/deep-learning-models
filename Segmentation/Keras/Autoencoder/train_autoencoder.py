import os
import keras
import tensorflow as tf
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import time
from keras.optimizers import Adam
from google.colab.patches import cv2_imshow
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from data_loader import load_data
from model.autoencoder import build_autoencoder

# 버전 출력
print(f'keras : {keras.__version__}')
print(f'tensorflow : {tf.__version__}')
print(f'numpy : {np.__version__}')
print(f'cv2 : {cv2.__version__}')
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Version : {torch.version.cuda}")
print(f"cudnn Version : {torch.backends.cudnn.version()}")
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"Count of GPUs : {torch.cuda.device_count()}")

# 데이터 불러오기
train_noisy, train_clean, test_noisy, test_clean = load_data()

# 모델 정의 및 학습
model = build_autoencoder()
model.summary()

model.compile(optimizer=Adam(0.001, beta_1=0.9), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_noisy, train_clean, batch_size=32, epochs=500, validation_split=0.1)

# 모델 저장
model.save('/content/drive/MyDrive/npy_friend/model_autoencoder.keras')

# 추론 및 저장
start_time = time.time()
imgs_test_pred = model.predict(test_noisy)
end_time = time.time()

np.save('/content/drive/MyDrive/npy_friend/imgs_test_pred_autoencoder.npy', imgs_test_pred)

# 추론 시간 출력
total_time = (end_time - start_time) * 1000
avg_time = total_time / len(test_noisy)
print(f"Total Inference Time: {total_time:.2f} ms")
print(f"Time per Inference Step (ms): {avg_time:.2f} ms")

# 시각화 함수
def display(array1, array2):
    n = 10
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(100, 20))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(256, 256))
        plt.gray()
        ax.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(256, 256))
        plt.gray()
        ax.axis('off')
    plt.show()

# 시각화 실행
display(test_noisy, imgs_test_pred)

# 수직 시각화
imgs_test_pred = np.squeeze(imgs_test_pred, axis=-1)
size = test_noisy.shape[0] // 5
w = test_noisy.shape[2]
h = test_noisy.shape[1]
t_img = np.zeros((3 * h, 5 * w), np.uint8)
for i in range(5):
    t_img[0:h, i * w:i * w + w] = 255 * test_noisy[i,:,:]
    t_img[h:h + h, i * w:i * w + w] = 255 * test_clean[i,:,:]
    t_img[h + h:h + h + h, i * w:i * w + w] = 255 * imgs_test_pred[i,:,:]
cv2_imshow(t_img)

# 손실 그래프
plt.plot(history.history['loss'], 'b+')
plt.plot(history.history['val_loss'], 'g+')
plt.title('Train and Validation loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Validation loss'], loc='upper right')
plt.show()

# 정량적 평가
psnr_value = peak_signal_noise_ratio(test_clean, imgs_test_pred)
ssim_loss = structural_similarity(test_clean, imgs_test_pred, channel_axis=-1, data_range=test_clean.max() - test_clean.min())
mse_value = mean_squared_error(test_clean.flatten(), imgs_test_pred.flatten())

print("**이미지 비교**")
print("PSNR :", psnr_value)
print("SSIM :", ssim_loss)
print("MSE :", mse_value)

# 평균 SSIM 계산
ssim_score = 0.0
for i in range(len(imgs_test_pred)):
    pred_img = imgs_test_pred[i]
    gt_img = test_clean[i]
    ssim_value = structural_similarity(gt_img, pred_img, data_range=gt_img.max() - gt_img.min(), win_size=3, channel_axis=-1)
    ssim_score += ssim_value
print(f"Mean SSIM1: {ssim_score / len(imgs_test_pred)}")
