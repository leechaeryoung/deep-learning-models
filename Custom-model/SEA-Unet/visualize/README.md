# Sobel Filter Output Visualization

This script visualizes the output of a 3x3 Sobel filter applied to an input image.
It is suitable for grayscale images such as ultrasound scans.


## 📄 File
- `visualize_sobel_output.py`: Runs the SobelConv2d module and shows the intermediate output of 4 edge channels:
  - Horizontal
  - Vertical
  - Diagonal ↘
  - Diagonal ↙

## 🛠 How to Use

1. Open `visualize_sobel_output.py`
2. Set the `image_path` to the location of your test image
3. Run the script

```bash
python visualize_sobel_output.py
```

The script will show the four Sobel-filtered outputs side by side.

## 📌 Note

- This is used as a helper visualization for SEA U-Net input preprocessing.
