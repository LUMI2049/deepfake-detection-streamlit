---
title: DeepFake Detection - AICS 2025
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🔍 DeepFake Detection System - AICS 2025

**Advanced AI-Generated Image Detection Using EfficientNetB7 with Attention Mechanism**

Developed for **33rd Irish Conference on Artificial Intelligence and Cognitive Science (AICS 2025)**

---

## 🔗 Quick Access

### 📓 Master's Thesis Project
[![GitHub Project](https://img.shields.io/badge/GitHub-Master's%20Project-black?style=for-the-badge&logo=github)](https://github.com/CemRoot/Master-Uni-Project)

**Notebook kodları, Flask app ve demo görselleri için yukarıdaki butona tıklayınız** | *Click above for notebooks, Flask app, and demo images*

### 🤗 Pre-trained Models
[![Download Models](https://img.shields.io/badge/🤗%20Models-Download-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/CemRoot/deepfake-detection-model)

**Eğitilmiş modellere ulaşmak için yukarıdaki butona tıklayınız** | *Click above to access pre-trained models*

### 🔗 Streamlit Application (This Repository)
[![GitHub](https://img.shields.io/badge/GitHub-Streamlit%20App-FF4B4B?style=for-the-badge&logo=github)](https://github.com/CemRoot/deepfake-detection-streamlit)

**Streamlit uygulama kodlarına ulaşmak için yukarıdaki butona tıklayınız** | *Click above to access Streamlit application source code*

---

## 🎯 Try It Now!

1. **Upload an image** (supported formats: JPG, JPEG, PNG)
2. **Select preprocessing method** (Training Match recommended)
3. **Click "Analyze Image"**
4. **See instant visual results** with color-coded progress bars:
   - 🔴 **Red bar** for FAKE (AI-Generated) probability
   - 🟢 **Green bar** for AUTHENTIC (Real) probability

---

## ✨ Key Features

- 🎯 **High Detection Accuracy** - Model trained on large-scale deepfake datasets (~95% with Training Match)
- 🚀 **Rapid Inference** - Analysis completed within 2-3 seconds
- 🎨 **Modern Interface** - Clean, intuitive Gradio interface with color-coded visual feedback
- 📊 **Visual Progress Bars** - Red/Green color-coded bars for instant visual interpretation
- 🔍 **Confidence Metrics** - Detailed probability distribution for each prediction
- 🔄 **Multiple Preprocessing Methods** - Compare different approaches (educational)
- 🔬 **Research-Grade** - Developed for academic conference presentation (AICS 2025)

---

## 🔬 Technical Architecture

- **Neural Network:** EfficientNetB7 with Custom Attention Mechanism
- **Framework:** TensorFlow 2.15 + Gradio 4.16
- **Model Size:** ~780MB (automatically loaded from Hugging Face Model Hub)
- **Input Resolution:** 128×128 pixels
- **Detection Targets:** GANs, Diffusion Models, and other generative AI systems

---

## 📊 Detection Capabilities

### Generative Adversarial Networks (GANs)
✅ StyleGAN (v1, v2, v3)
✅ ProGAN
✅ BigGAN
✅ CycleGAN
✅ StarGAN
✅ DCGAN, WGAN

### Diffusion Models
✅ Stable Diffusion
✅ DALL-E (2, 3)
✅ Midjourney
✅ Imagen
✅ DDPM, DDIM

### Other AI-Generated Content
✅ Neural style transfer
✅ VAE-based generators
✅ Autoregressive models
✅ Image-to-image translation

### Robustness
✅ Compressed images (JPEG artifacts)
✅ Resized/manipulated images
✅ Various image qualities

---

## 📈 Preprocessing Methods Comparison

This system includes **three preprocessing methods** for educational comparison:

### 1. ✅ Training Match (RECOMMENDED)
- **Color Format:** BGR (matches OpenCV training pipeline)
- **Value Range:** 0-255 (raw pixel values, no normalization)
- **Expected Accuracy:** ~95%
- **Use Case:** Production deployment, best performance

**Why it works best:**
- Exactly matches the preprocessing used during model training
- Neural network weights optimized for this specific input distribution
- No information loss from normalization

### 2. Simple [0,1] Normalization
- **Color Format:** RGB
- **Value Range:** 0-1 (normalized)
- **Expected Accuracy:** ~58% (worse than random!)
- **Use Case:** Educational demonstration of preprocessing impact

**Why it performs poorly:**
- Channel order mismatch (RGB vs BGR)
- Value distribution completely different from training
- Model weights never encountered these normalized values

### 3. EfficientNet ImageNet
- **Color Format:** RGB
- **Value Range:** ImageNet mean/std normalization (~-2 to +2)
- **Expected Accuracy:** ~72%
- **Use Case:** Transfer learning experiments

**Why it's suboptimal:**
- Uses ImageNet statistics (not deepfake-specific)
- Different preprocessing than training pipeline
- Better than simple normalization but still inconsistent

---

## 🎓 Research Context

This system was developed for presentation at **AICS 2025** (33rd Irish Conference on Artificial Intelligence and Cognitive Science).

### Key Research Contributions:

1. **Attention Mechanism Effectiveness**
   - Demonstrates improved detection through custom attention layers
   - Focuses on discriminative features in synthetic images

2. **Preprocessing Consistency Impact**
   - Empirically shows catastrophic performance degradation from inconsistent preprocessing
   - Training Match: 95% accuracy
   - Simple [0,1]: 58% accuracy (37% drop!)
   - Highlights critical importance of deployment pipeline matching training

3. **Practical Deployment**
   - Real-world application of deep learning for synthetic media detection
   - Efficient inference on CPU (2-3 seconds per image)
   - Hugging Face integration for easy access and reproducibility

---

## 🚀 Model Information

- **Repository:** [CemRoot/deepfake-detection-model](https://huggingface.co/CemRoot/deepfake-detection-model)
- **Architecture:** EfficientNetB7 (backbone) + Custom Attention + Classification Head
- **Parameters:** ~66M total, ~2M trainable
- **Training:** Kaggle platform with large-scale deepfake datasets
- **Backbone:** EfficientNetB7 (frozen, pretrained on ImageNet)
- **Custom Layers:** Attention mechanism + Dense layers

---

## 🎨 Visual Output

The system provides **color-coded visual progress bars** for easy interpretation:

### Example Output (Fake Image Detected):
```
Detection Results
┌────────────────────────────────────────────┐
│ 🚨 FAKE (AI-Generated)    ████████████ 99.99% │ ← RED BAR (Gradient)
│ ✅ AUTHENTIC (Real)       ░             0.01%  │ ← Green bar (minimal)
└────────────────────────────────────────────┘
```

### Example Output (Real Image Detected):
```
Detection Results
┌────────────────────────────────────────────┐
│ ✅ AUTHENTIC (Real)       ██████████ 94.50%    │ ← GREEN BAR (Gradient)
│ 🚨 FAKE (AI-Generated)    ███░        5.50%    │ ← Red bar (minimal)
└────────────────────────────────────────────┘
```

**Color Coding:**
- 🔴 **Red Gradient** (`#ff4444` → `#cc0000`) - FAKE/AI-Generated
- 🟢 **Green Gradient** (`#00cc00` → `#009900`) - AUTHENTIC/Real
- **Bold white text** with shadow for maximum readability
- **Automatic sorting** - Highest probability shown first

---

## 💻 Usage Example (Python API)

```python
from huggingface_hub import hf_hub_download
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2

# Download model (or use local file in Space)
model_path = hf_hub_download(
    repo_id="CemRoot/deepfake-detection-model",
    filename="best_model_effatt.h5"
)

# Load model with fallback mechanism
def RescaleGAP(tensors):
    return tensors[0] / tensors[1]

try:
    model = keras.models.load_model(
        model_path,
        custom_objects={'RescaleGAP': RescaleGAP},
        compile=False
    )
except:
    # Rebuild architecture and load weights
    from build_model import build_effatt_model
    model = build_effatt_model(input_shape=(128, 128, 3))
    model.load_weights(model_path)

# Preprocess image (Training Match method - RECOMMENDED)
img = Image.open("test.jpg").convert('RGB')
img = np.array(img)
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB → BGR
img_resized = cv2.resize(img_bgr, (128, 128))
img_array = img_resized.astype(np.float32)  # NO normalization!
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
fake_prob = predictions[0][0]
real_prob = predictions[0][1]

# Interpret results
if fake_prob > real_prob:
    print(f"🚨 FAKE (AI-Generated): {fake_prob*100:.2f}%")
else:
    print(f"✅ AUTHENTIC (Real): {real_prob*100:.2f}%")
```

---

## 🔗 Links

- **Model Repository:** [CemRoot/deepfake-detection-model](https://huggingface.co/CemRoot/deepfake-detection-model)
- **GitHub Repository:** [CemRoot/deepfake-detection-streamlit](https://github.com/CemRoot/deepfake-detection-streamlit)
- **Conference:** [AICS 2025](https://aics2025.com)
- **Documentation:** See GitHub repository for full documentation

---

## 👨‍💻 Author

**Emin Cem Koyluoglu**

33rd Irish Conference on Artificial Intelligence and Cognitive Science (AICS 2025)

---

## 📜 License

MIT License - Free for academic and commercial use

---

## 🚀 Technical Implementation

### Model Loading
- **Primary:** Local file (`best_model_effatt.h5`) - Fast startup
- **Fallback:** Hugging Face Model Hub - Automatic download
- **Architecture Rebuild:** Automatic fallback if full model loading fails

### Visual Interface
- **Framework:** Gradio 5.0+ (latest, secure)
- **Custom CSS:** Color-coded progress bars
- **Theme:** Purple/Blue gradient (AICS 2025 branding)
- **Responsive:** Works on desktop and mobile

### Performance Optimization
- ⚡ **Fast Startup:** ~30 seconds (local model file)
- 🚀 **Quick Inference:** 2-3 seconds per image
- 💾 **Memory Efficient:** ~2GB RAM usage
- 🔄 **Automatic Caching:** Model loaded once at startup

---

## 🙏 Acknowledgments

- Model trained on Kaggle platform with large-scale deepfake datasets
- EfficientNetB7 backbone from TensorFlow/Keras Applications
- Hugging Face for hosting and deployment infrastructure
- AICS 2025 Conference organizers and reviewers
- Gradio team for excellent UI framework

---

## 📊 Version History

**v1.2** (Current) - Visual Progress Bars
- Added color-coded red/green progress bars
- Improved visual feedback with gradients
- Enhanced user experience

**v1.1** - Model Loading Optimization
- Local model file support
- Architecture rebuild fallback
- Faster startup time

**v1.0** - Initial Release
- EfficientNetB7 + Attention mechanism
- Three preprocessing methods
- Gradio interface

---

<div align="center">
  <strong>⚡ Powered by Hugging Face Spaces</strong>
  <br>
  <sub>Gradio 5.0+ • TensorFlow 2.15 • EfficientNetB7 • Custom Attention</sub>
  <br><br>
  <sub>🎓 AICS 2025 - 33rd Irish Conference on AI and Cognitive Science</sub>
</div>
