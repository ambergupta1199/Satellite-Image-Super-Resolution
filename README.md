### Group Project(Course CS5546 - Introduction to Agriculture Cyber Physical Systems)
- My role is to train ESRGAN model on our dataset and compare the super-reolved image with high resolution ground truth image using evaluation metrics like PSNR, SSIM.
- I also worked on classification of Satellite images both low resolution and high resolution based on their NDVI(Normalized Difference Vegetation Index) value and then used these images as dataset for Trained ESRGAN model to see which type of vegetation provides good result on which model.
# 🌍 Satellite Image Super-Resolution

Enhancing satellite image resolution and classifying vegetation types for precision agriculture and remote sensing using state-of-the-art deep learning models. This project focuses on super-resolving multispectral satellite imagery and evaluating performance across land cover types.

---

## 🚀 Project Objectives

- 🧠 Test deep learning-based super-resolution models tailored for multispectral satellite images.
- 🗂️ Utilize a dataset of **30,000 paired low- and high-resolution agricultural images** for training and evaluation.
- 🎯 Preserve **spectral accuracy** while improving **spatial resolution**.
- 🔍 Identify the most effective super-resolution techniques through comparative analysis.
- 🏗️ Train existing architectures (e.g., SRCNN, ESRGAN, SwinIR) and evaluate their performance.
- 🌐 Propose a novel **GAN architecture** combining **image captions + low-resolution inputs** to generate high-resolution outputs.
- 🌱 Classify images based on **NDVI values** into:  
  - 🏜️ Barren land/desert  
  - 🌾 Sparse vegetation  
  - 🌽 Moderate vegetation (cropland)  
  - 🌳 Dense vegetation/forest  
- 📊 Analyze model performance across different land types.

---

## 🛠️ Methodology

### 🔧 Model Architecture
- Used deep learning models: **SRCNN**, **ESRGAN**, **SwinIR**, and a custom **Multimodal GAN**.
- Integrated **spectral information** from multiple channels.
- Designed a new **GAN-based architecture** leveraging both image captions and visual data.

### 📈 Training Strategy
- Supervised learning with **high-resolution ground truth**.
- Loss functions: **Mean Squared Error (MSE)**, **Perceptual Loss**, **Adversarial Loss**.
- Optimized using **Adam** with a learning rate scheduler.

### 🧪 Evaluation Metrics
- 🔢 **PSNR (Peak Signal-to-Noise Ratio)** – Measures image quality.
- 🧠 **SSIM (Structural Similarity Index)** – Evaluates perceptual similarity.
- 🌈 **SAM (Spectral Angle Mapper)** – Assesses spectral fidelity.

---

## 📦 Dataset

- 📸 **30,000 paired** low- and high-resolution satellite images.
- 🌾 Agricultural regions with varying vegetation types.
- 🌐 Multispectral bands for enhanced classification and super-resolution.

---

## ✅ Expected Outcomes

- 📷 High-resolution satellite images generated from low-resolution inputs.
- 📉 Quantitative improvement in resolution and visual quality over traditional interpolation.
- 🌍 Reliable classification of vegetation cover using NDVI indices.
- 🧪 Performance comparison of top models across different land cover types.

---

## 📌 Technologies Used

- 🐍 Python
- 🧠 PyTorch / TensorFlow
- 📊 NumPy, Pandas, Matplotlib
- 🌐 OpenCV, Scikit-learn
- 🛰️ Remote sensing libraries (e.g., rasterio, GDAL)


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- Sentinel and PLanetScope satellite datasets
- Contributors to SRCNN, ESRGAN, SwinIR
- Remote sensing and geospatial open communities

---

⭐ If you find this project helpful, feel free to give it a **star** and share it with others!
