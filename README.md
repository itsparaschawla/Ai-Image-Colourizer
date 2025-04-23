# Ai-Image-Colourizer
This project implements a deep learning-based approach to automatically colorize black and white (grayscale) images using a U-Net model trained on the Places365 dataset. The project also includes an interactive Streamlit web app for easy image uploads, real-time predictions, and evaluation using SSIM and PSNR metrics.

# 📌 Features
1. Upload grayscale (B/W) images 
2. Colorize them using a trained U-Net model (128 x 128 img size)
3. View SSIM (Structural Similarity Index) and PSNR (Peak Signal-to-Noise Ratio) for quality comparison
4. Download the colorized result
5. All in a clean, browser-based UI built with Streamlit

# 🧠 Model Architecture
The model used is a U-Net architecture, which is effective for image-to-image translation tasks. It was trained using:
  ->Input: Lightness (L) channel from the Lab color space
  ->Output: Color channels (a and b)

# 📂 Dataset
The model was trained on a subset* of the Places365 dataset, which contains over 1.8 million images spanning 365 scene categories like mountains, forests, cityscapes, and more. It provides a diverse and rich variety of environmental context, ideal for learning realistic colorization

Original Dataset Link: https://www.kaggle.com/datasets/benjaminkz/places365 \
*Subset Dataset Link: https://drive.google.com/drive/folders/1SCd1MvrHPpFX5jHySAKZfVure5XGhEMX?usp=sharing \
*(Subset contains only images of mountains(5000) and forests(5000), total 10000 images split into 90:5:5 (train:test:val) for model training)\
*(it contains .tar.gz files - need to be extracted)

# 🚀 Run the Bild
Make sure you have Python installed, then ->\
Open the build folder in your code editor and then ->\
pip install -r requirements.txt\
streamlit run app.py