
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_squared_error
import numpy as np

def compute_metrics(gt, pred):
    psnr_value = peak_signal_noise_ratio(gt, pred)
    ssim_value = structural_similarity(gt, pred, channel_axis=-1, data_range=gt.max()-gt.min())
    mse_value = mean_squared_error(gt.flatten(), pred.flatten())
    return psnr_value, ssim_value, mse_value


psnr_value, ssim_value, mse_value = compute_metrics(imgs_mask_test, imgs_test_pred)

print("PSNR :", psnr_value)
print("SSIM :", ssim_value)
print("MSE :", mse_value)

