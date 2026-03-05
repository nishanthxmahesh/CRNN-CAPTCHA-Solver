CAPTCHA Recognition with CRNN & Explainable AI
This project implements a deep learning pipeline to solve 5-character alphanumeric CAPTCHAs using a Convolutional Recurrent Neural Network (CRNN). It features advanced evaluation metrics and Explainable AI (Grad-CAM) to visualize model decisions.

🚀 Features
Dual Generators: Includes both a standard library-based generator and a custom high-noise generator (lines, dots, blur).
Architecture: CRNN (CNN for feature extraction + Bidirectional LSTM for sequence modeling).
Data Augmentation: Real-time brightness and contrast adjustments using tf.data pipelines.
Explainable AI: Grad-CAM heatmaps to show which image regions triggered specific predictions.
Advanced Metrics: Macro-averaged F1-scores and per-position confusion matrices.
🛠️ Project Structure
Data Generation: Scripts to create 100k+ synthetic CAPTCHA images.
Training: High-performance TensorFlow pipeline with EarlyStopping and Learning Rate reduction.
Evaluation: Detailed analysis of character-level accuracy.
Inference & XAI: Tools to predict new samples and explain them via heatmaps.
📦 Requirements
TensorFlow 2.x
Pillow (PIL)
OpenCV (cv2)
Matplotlib / Seaborn
Scikit-learn
📊 Model Architecture
Input: 60x160 RGB or 70x200 Grayscale images.
CNN: 4 Conv blocks with Batch Normalization.
RNN: Bidirectional LSTM (128 units).
Output: Dense Softmax layer reshaped to (5, 36) classes.
