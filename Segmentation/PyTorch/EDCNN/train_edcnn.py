
import torch
import numpy as np

# 모델 초기화
model = EDCNN(in_ch=1, out_ch=32, sobel_ch=32)
criterion = CompoundLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

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



# 모델 학습 및 검증 수행
train_losses, val_losses = train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=500, device=device)

# 학습 완료 후 모델 저장
torch.save(model.state_dict(), "/content/drive/MyDrive/npy_friend/mode_EDCNN.pth")
print("-------모델이 저장되었습니다.-------")

# 손실 값 시각화 (matplotlib 사용)
import matplotlib.pyplot as plt

epochs_range = range(1, len(train_losses) + 1)
plt.plot(epochs_range, train_losses, 'b+', label='Train loss')
plt.plot(epochs_range, val_losses, 'g+', label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.title('Train and Validation loss')

plt.show()
