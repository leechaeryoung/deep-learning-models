
import torch
import numpy as np

def predict_and_save(model, test_loader, save_path, device='cuda'):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for noisy_img, _ in test_loader:
            noisy_img = noisy_img.to(device)
            output = model(noisy_img)
            output = output.cpu().numpy()
            all_preds.append(output)
    all_preds_np = np.concatenate(all_preds, axis=0)
    np.save(save_path, all_preds_np)
    print(f"Predictions saved to {save_path}")
