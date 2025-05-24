
import torch
import numpy as np

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    train_losses = np.zeros((num_epochs), dtype=np.float32)
    val_losses = np.zeros((num_epochs), dtype=np.float32)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for noisy_img, clean_img in train_loader:
            noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)
            output = model(noisy_img)
            loss = criterion(output, clean_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses[epoch] = avg_train_loss

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for noisy_img, clean_img in val_loader:
                noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)
                output = model(noisy_img)
                loss = criterion(output, clean_img)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses[epoch] = avg_val_loss

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses
