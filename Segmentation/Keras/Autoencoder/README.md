# Autoencoder

This repository provides a Keras implementation of a basic Autoencoder model for speckle noise removal in ultrasound images.

## Model Overview

- Input shape: (256, 256, 1)
- Encoder: 2 convolutional layers with max pooling
- Decoder: 2 transposed convolutional layers
- Output: Single-channel image with sigmoid activation
- Loss: Binary Crossentropy
- Optimizer: Adam (learning rate 0.001)

## Features

- Loads training and test data from `.npy` files
- Trains the model and saves the result in `.keras` format
- Predicts test data and saves the output in `.npy` format
- Measures inference time (total and average)
- Visualizes predictions vs. ground truth
- Calculates evaluation metrics (PSNR, SSIM, MSE)
- Displays training and validation loss graph


## How to Run

Make sure the following data files exist (in `.npy` format):

- train_imgs.npy
- train_mask.npy
- test_imgs.npy
- test_mask.npy

Set the correct path in `data_loader.py` if different from default.

Then run:

```bash
python train_autoencoder.py
```

## Note

The input/output paths must be customized in `data_loader.py` according to your environment (e.g., Google Drive, local).

