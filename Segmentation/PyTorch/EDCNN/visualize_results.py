
import cv2
import numpy as np
import torch
from google.colab.patches import cv2_imshow

def visualize(model, imgs_test_tensor, imgs_mask_test_tensor, device='cuda'):
    model.eval()
    with torch.no_grad():
        images = []
        for i in range(5):
            noisy_img = imgs_test_tensor[i].unsqueeze(0).to(device)
            clean_img = imgs_mask_test_tensor[i].cpu().numpy().squeeze()
            output = model(noisy_img)
            output_img = output[0].cpu().detach().numpy().squeeze()

            clean_img_cv = np.uint8(clean_img * 255)
            denoised_img_cv = np.uint8(output_img * 255)
            noisy_img_cv = np.uint8(noisy_img.cpu().numpy().squeeze() * 255)

            clean_img_cv = cv2.cvtColor(clean_img_cv, cv2.COLOR_GRAY2BGR)
            denoised_img_cv = cv2.cvtColor(denoised_img_cv, cv2.COLOR_GRAY2BGR)
            noisy_img_cv = cv2.cvtColor(noisy_img_cv, cv2.COLOR_GRAY2BGR)

            column = np.vstack([clean_img_cv, denoised_img_cv, noisy_img_cv])
            images.append(column)

        final_image = np.hstack(images)
        cv2.putText(final_image, 'Ground Truth', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(final_image, 'Denoised Image', (0, final_image.shape[0] // 3 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(final_image, 'Noisy Image', (0, 2 * final_image.shape[0] // 3 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2_imshow(final_image)
