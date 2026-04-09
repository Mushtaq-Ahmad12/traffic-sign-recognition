# Traffic Sign Recognition

A high-performance deep learning system designed to classify 43 different types of traffic signs using Convolutional Neural Networks (CNNs). Built with PyTorch and Keras 3.

## 🚀 Features
- **43-Class Classification**: Recognizes everything from speed limits to construction signs.
- **Robust CNN Architecture**: Implements batch normalization, dropout, and multiple convolutional blocks for high accuracy.
- **Real-time Inference**: Includes a modern web interface built with Gradio for easy testing.
- **Data Augmentation**: Robust training pipeline with random rotations and color jittering.

## 📂 Project Structure
```text
├── data/               # GTSRB Dataset
├── logs/               # Training logs and confusion matrix
├── models/             # Saved model checkpoints
├── scripts/            # Utility scripts (download, etc.)
└── src/                # Core implementation
    ├── app.py          # Gradio Web Interface
    ├── evaluate.py     # Evaluation and metrics
    ├── model.py        # CNN Architecture
    ├── preprocessing.py # Data loading and transforms
    └── train.py        # Training pipeline
```

## 🛠️ Installation
1. Install dependencies:
   ```bash
   pip install torch torchvision gradio scikit-learn seaborn matplotlib tqdm
   ```
2. Download data:
   ```bash
   python scripts/download_data.py
   ```
3. Train the model:
   ```bash
   python -m src.train
   ```

## 🧪 Evaluation
Run `python -m src.evaluate` to generate the classification report and confusion matrix.

## 🌐 Web App
Run `python -m src.app` to launch the interactive web interface.
