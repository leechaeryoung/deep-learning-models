# Skip Autoencoder

This repository provides a Keras implementation of the Skip Autoencoder for speckle noise removal in ultrasound images.

## Model Overview

- Architecture: Autoencoder with skip connections
- Input size: (256, 256, 1)
- Encoder: 6-layer convolutional blocks
- Decoder: 6-layer transpose convolution with skip connections
- Latent bottleneck: Dense(128)
- Activation: LeakyReLU, Sigmoid
- Loss: Binary Crossentropy
- Optimizer: Adam (lr=0.001)

## Features

- Custom skip connection between encoder and decoder
- Visualization of input, denoised output, and ground truth
- Evaluation metrics: PSNR, SSIM, MSE, mean SSIM
- Inference speed measurement included



## Usage

Train the model and evaluate:

```bash
python train_skipauto.py
```
Input/output file paths must be individually specified in data_loader.py and train_skipauto.py depending on your local or cloud environment.
