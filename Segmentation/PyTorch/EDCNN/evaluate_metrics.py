import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_squared_error
import numpy as np

#시간추
model.eval()
start_time = time.time()
total_images = 0

with torch.no_grad():
    for noisy_img, clean_img in test_loader:
        noisy_img = noisy_img.to(device)
        output = model(noisy_img)
        total_images += noisy_img.size(0)

end_time = time.time()

total_time = (end_time - start_time) * 1000
average_time_per_inference = total_time / total_images if total_images > 0 else 0

print(f"Total Inference Time: {total_time:.2f} ms")
print(f"Time per Inference Step (ms): {average_time_per_inference:.2f} ms")



def compute_metrics(gt, pred):
    psnr_value = peak_signal_noise_ratio(gt, pred)
    ssim_value = structural_similarity(gt, pred, channel_axis=-1, data_range=gt.max()-gt.min())
    mse_value = mean_squared_error(gt.flatten(), pred.flatten())
    return psnr_value, ssim_value, mse_value


psnr_value, ssim_value, mse_value = compute_metrics(imgs_mask_test, imgs_test_pred)

print("PSNR :", psnr_value)
print("SSIM :", ssim_value)
print("MSE :", mse_value)


from skimage.metrics import structural_similarity as ssim
import numpy as np

ssim_score = 0.0
for i in range(len(imgs_test_pred)):
    pred_img = imgs_test_pred[i]
    gt_img = imgs_mask_test[i]
    ssim_value = ssim(gt_img, pred_img, data_range=gt_img.max() - gt_img.min(), win_size=3, channel_axis=-1)
    ssim_score += ssim_value

mean_ssim1 = ssim_score / len(imgs_test_pred)
print(f"Mean SSIM1: {mean_ssim1}")


