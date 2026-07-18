# 🧠 Emotion CNN Activation Maps Visualizer

Analysis system 

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Face%20Data/Dockerfile)

> An interactive **Explainable AI (XAI)** tool for visualizing CNN activation maps in facial emotion recognition, enabling researchers and developers to understand what neural networks "see" when classifying emotions.

![Activation Maps Demo](Face%20Data/outputs/class_activation_map.png)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Training](#-model-training)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project addresses the **black-box problem** in deep learning by providing visual explanations for CNN-based emotion classification. Understanding *why* a model makes certain predictions is crucial for:

- 🎓 **Education** - Teaching neural network concepts visually
- 🔬 **Research** - Debugging and improving model architectures
- ✅ **Validation** - Ensuring models focus on relevant facial features
- 🏥 **Healthcare AI** - Building trust in AI-assisted emotion analysis

### Supported Emotion Classes

| Emotion | Description |
|---------|-------------|
| 😠 Angry | Facial expressions indicating anger |
| 🤢 Disgust | Expressions of disgust or revulsion |
| 😨 Fear | Fearful or scared expressions |
| 😊 Happy | Smiling or joyful expressions |
| 😐 Neutral | Neutral or expressionless faces |
| 😢 Sad | Sad or sorrowful expressions |
| 😲 Surprise | Surprised or shocked expressions |

---

## ✨ Key Features

### Visualization Techniques

| Feature | Description |
|---------|-------------|
| **Class Activation Maps (CAM)** | Highlights image regions most relevant to the predicted class |
| **Grad-CAM** | Gradient-weighted class activation for finer interpretability |
| **Layer-wise Activations** | Visualize filter responses at any convolutional layer |
| **Filter Exploration** | Inspect individual filters (4-32 configurable) |

### Application Features

| Feature | Description |
|---------|-------------|
| **Interactive Web UI** | Modern Streamlit interface with real-time updates |
| **Custom Model Support** | Load your own trained PyTorch models |
| **Demo Mode** | Instant testing without uploading images |
| **Download Overlays** | Export activation heatmaps for reports/papers |
| **Layer Selection** | Choose specific layers to visualize |
| **GPU Acceleration** | CUDA support for faster inference |

---

## 🎬 Demo

### Quick Start Demo

```bash
# Clone and run in under 2 minutes
git clone https://github.com/0011Ashwin/Emotion-CNN-Activation.git
cd Emotion-CNN-Activation/Face\ Data
pip install -r requirements.txt
streamlit run app.py
```

Then click **"Demo Mode"** in the sidebar to see activation maps instantly!

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Image Upload│  │ Model Config│  │ Visualization Controls  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ActivationMapVisualizer                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  EmotionCNN (ResNet18-based)                                ││
│  │  ├── Conv Layers with Forward Hooks                         ││
│  │  ├── Activation Storage (OrderedDict)                       ││
│  │  └── FC Layer (7 emotion classes)                           ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐    │
│  │ Preprocess  │→ │ Inference   │→ │ CAM/Grad-CAM Gen      │    │
│  └─────────────┘  └─────────────┘  └───────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output Generation                           │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐    │
│  │ Heatmap     │  │ Overlay     │  │ Layer Activation Grid │    │
│  │ Generation  │  │ Composition │  │ Visualization         │    │
│  └─────────────┘  └─────────────┘  └───────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Model Architecture

```
ResNet18 (Modified for Emotion Recognition)
├── Conv1 (7×7, 64 filters)
├── BatchNorm + ReLU + MaxPool
├── Layer1 (2× BasicBlock, 64 filters)  ← Activation hooks
├── Layer2 (2× BasicBlock, 128 filters) ← Activation hooks
├── Layer3 (2× BasicBlock, 256 filters) ← Activation hooks
├── Layer4 (2× BasicBlock, 512 filters) ← Activation hooks + CAM
├── AdaptiveAvgPool2d
└── FC (512 → 7 emotion classes)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU acceleration)

### Method 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/0011Ashwin/Emotion-CNN-Activation.git
cd Emotion-CNN-Activation/Face\ Data

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Docker Installation

```bash
# Build the Docker image
cd Face\ Data
docker build -t emotion-cnn-activation .

# Run the container
docker run -p 8501:8501 emotion-cnn-activation
```

### Method 3: Conda Environment

```bash
cd Face\ Data
conda env create -f environment.yml
conda activate emotion-cnn
```

---

## 💻 Usage

### Running the Web Application

```bash
cd Face\ Data
streamlit run app.py
```

The app opens at `http://localhost:8501`

### Using the Interface

1. **Model Settings** (Sidebar)
   - Specify custom model path or use pre-trained weights
   
2. **Visualization Controls** (Sidebar)
   - Adjust number of filters (4-32)
   - Select specific layers to visualize
   - Toggle Demo Mode

3. **Upload & Analyze**
   - Upload a face image (JPG/PNG)
   - Or click "Demo Mode" for instant testing

4. **Explore Results**
   - View predicted emotion with confidence
   - Examine Class Activation Map overlay
   - Expand layers to see individual filter activations
   - Download overlays for reports

### Command-Line Usage

```python
from visualize_activations import ActivationMapVisualizer

# Initialize visualizer
visualizer = ActivationMapVisualizer(
    model_path="model/emotion_model.pth",
    img_size=(224, 224)
)

# Visualize activations
predicted_class = visualizer.visualize_activation_maps(
    img_path="path/to/face.jpg",
    output_dir="outputs/",
    num_filters=16
)

print(f"Predicted: {visualizer.emotion_labels[predicted_class]}")
```

---

## 🎯 Model Training

### Training Your Own Model

```bash
cd Face\ Data
python train_emotion_model.py \
    --train_dir train/ \
    --val_dir test/ \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001
```

### Dataset Structure

```
Face Data/
├── train/
│   ├── angry/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── happy/
│   ├── sad/
│   └── ... (7 emotion folders)
└── test/
    ├── angry/
    ├── happy/
    └── ...
```

### Training Features

- ✅ Data augmentation (rotation, flip, color jitter)
- ✅ Learning rate scheduling
- ✅ Early stopping
- ✅ Training/validation metrics visualization
- ✅ Best model checkpointing

---

## 📁 Project Structure

```
Emotion-CNN-Activation/
├── README.md                      # Project documentation
├── pyproject.toml                 # Python project configuration
│
└── Face Data/                     # Main application directory
    ├── app.py                     # Streamlit web application
    ├── visualize_activations.py   # Core visualization logic & model
    ├── train_emotion_model.py     # Model training script
    ├── requirements.txt           # Python dependencies
    ├── Dockerfile                 # Docker configuration
    ├── environment.yml            # Conda environment
    ├── azure-webapp.yaml          # Azure deployment config
    │
    ├── model/                     # Trained model weights
    │   └── emotion_model.pth
    │
    ├── outputs/                   # Generated visualizations
    │   ├── class_activation_map.png
    │   ├── activation_map_layer1_0_conv1.png
    │   └── ... (layer activations)
    │
    ├── train/                     # Training dataset
    │   └── {emotion}/             # Images per emotion
    │
    └── test/                      # Test dataset
        └── {emotion}/
```

---

## 🔬 Technical Details

### Technologies Used

| Category | Technologies |
|----------|--------------|
| **Deep Learning** | PyTorch 2.0+, TorchVision |
| **Model Architecture** | ResNet18 (Transfer Learning) |
| **Visualization** | OpenCV, Matplotlib, NumPy |
| **Web Framework** | Streamlit |
| **Image Processing** | Pillow, OpenCV |
| **Deployment** | Docker, Azure Web Apps |

### How Activation Maps Work

1. **Forward Hooks**: Register hooks on convolutional layers to capture intermediate activations
2. **Activation Extraction**: During inference, activations are stored in an OrderedDict
3. **Normalization**: Activations are normalized to [0, 1] range
4. **Heatmap Generation**: Apply colormap (JET) to create visual heatmaps
5. **Overlay Composition**: Blend heatmap with original image (50% opacity)

### CAM vs Grad-CAM

| Aspect | CAM | Grad-CAM |
|--------|-----|----------|
| Requires GAP layer | Yes | No |
| Uses gradients | No | Yes |
| Resolution | Lower | Higher |
| Model modification | Required | Not required |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/NewVisualization`)
3. **Commit** your changes (`git commit -m 'Add attention visualization'`)
4. **Push** to the branch (`git push origin feature/NewVisualization`)
5. **Open** a Pull Request

### Ideas for Contributions

- [ ] Add attention mechanism visualization
- [ ] Support for other architectures (VGG, EfficientNet)
- [ ] Real-time webcam emotion detection
- [ ] Batch processing mode
- [ ] Export visualizations as video

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch** team for the excellent deep learning framework
- **Streamlit** for the intuitive web app framework
- **FER2013** dataset creators for emotion recognition data
- Research papers on CAM and Grad-CAM visualization techniques

---

## 📚 References

- Zhou et al., "Learning Deep Features for Discriminative Localization" (CAM)
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
- He et al., "Deep Residual Learning for Image Recognition" (ResNet)

---

<p align="center">
  Built with ❤️ for Explainable AI
  <br>
  <a href="https://github.com/0011Ashwin">@0011Ashwin</a>
</p>
