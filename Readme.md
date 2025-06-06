# 🖼️ Image Quality Assessment (IQA) System - MainFiles

## 📖 Overview

This folder contains the core implementation of a comprehensive Image Quality Assessment (IQA) system that evaluates and compares image quality using 12 different metrics. The system is designed for researchers, developers, and practitioners working in computer vision, image processing, and perceptual quality assessment.

## 🎯 What This System Does

The IQA system provides:
- **12 State-of-the-art IQA metrics** including SSIM, MS-SSIM, PSNR, LPIPS variants, FSIM, VSI, GMSD, BRISQUE, NIQE, and CLIP-IQA
- **Dataset loading utilities** for BAPPS and LIVE-MD datasets
- **Batch evaluation capabilities** for large-scale image quality assessment
- **Comparative analysis tools** to evaluate metric performance
- **Interactive Jupyter notebook** for exploratory analysis

## 🗂️ Core Files Structure

### 📊 Main Engine
- **[`metric.py`](metric.py)** - Core IQA metrics implementation with 12 different quality metrics
- **[`dataset_loaders.py`](dataset_loaders.py)** - Dataset loading utilities for BAPPS, LIVE-MD, and custom datasets

### 📋 Dataset Files
- **[`data.ipynb`](data.ipynb)** - This file help to download the BAPPS and LIVE_MD datasets.

### 📈 Analysis Tools
- **Interactive analysis capabilities** for metric comparison and evaluation
- **Batch processing support** for large image datasets
- **Statistical analysis tools** for metric performance evaluation

## 🚀 Quick Start

### 1. **Basic Metric Evaluation**
```python
from metric import IQAMetrics

# Initialize the IQA system
iqa = IQAMetrics(device='cpu', target_size=(256, 256))

# Evaluate image quality
score = iqa.evaluate_pair('reference.jpg', 'distorted.jpg', 'ssim')
print(f"SSIM Score: {score}")
```

### 2. **Multiple Metrics Evaluation**
```python
# Evaluate multiple metrics at once
metrics = ['ssim', 'psnr', 'lpips', 'ms_ssim']
results = {}

for metric in metrics:
    score = iqa.evaluate_pair('ref.jpg', 'dist.jpg', metric)
    results[metric] = score
    
print("Quality Assessment Results:", results)
```

### 3. **Dataset Loading**
```python
from dataset_loaders import DatasetLoader

# Load BAPPS dataset samples
loader = DatasetLoader()
samples = loader.load_bapps_dataset(num_samples=100)

# Load LIVE-MD dataset
live_data = loader.load_live_md_dataset()
```

### 4. **Batch Evaluation**
```python
# Evaluate multiple image pairs
image_pairs = [
    ('ref1.jpg', 'dist1.jpg'),
    ('ref2.jpg', 'dist2.jpg'),
    # ... more pairs
]

results = []
for ref, dist in image_pairs:
    score = iqa.evaluate_pair(ref, dist, 'ssim')
    results.append(score)
```

## 📊 Available IQA Metrics

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| **SSIM** | Full-Reference | Structural Similarity Index | [0, 1] |
| **MS-SSIM** | Full-Reference | Multi-Scale SSIM | [0, 1] |
| **PSNR** | Full-Reference | Peak Signal-to-Noise Ratio | [0, ∞) |
| **LPIPS (Alex)** | Full-Reference | Learned Perceptual Image Patch Similarity | [0, 1] |
| **LPIPS (VGG)** | Full-Reference | LPIPS with VGG backbone | [0, 1] |
| **LPIPS (SqueezeNet)** | Full-Reference | LPIPS with SqueezeNet | [0, 1] |
| **FSIM** | Full-Reference | Feature Similarity Index | [0, 1] |
| **VSI** | Full-Reference | Visual Saliency Index | [0, 1] |
| **GMSD** | Full-Reference | Gradient Magnitude Similarity Deviation | [0, ∞) |
| **BRISQUE** | No-Reference | Blind/Referenceless Image Spatial Quality | [0, 100] |
| **NIQE** | No-Reference | Natural Image Quality Evaluator | [0, ∞) |
| **CLIP-IQA** | No-Reference | CLIP-based Image Quality Assessment | [0, 1] |

## 📋 Dataset Information

### LIVE-MD Dataset
- **Location**: [`LIVE_MD/LIVE_MD.txt`](LIVE_MD/LIVE_MD.txt)
- **Format**: CSV with columns: `dis_img_path`, `dis_type`, `ref_img_path`, `score`
- **Content**: 450+ image pairs with multiply distorted images
- **Distortion Types**: Blur + JPEG, Blur + Noise combinations
- **Usage**: Benchmark dataset for IQA metric evaluation

### BAPPS Dataset Support
- **Purpose**: Berkeley-Adobe Perceptual Patch Similarity dataset
- **Usage**: Human perceptual similarity judgments
- **Integration**: Loaded via `dataset_loaders.py`

## 🔧 System Requirements

### Dependencies
- Python 3.7+
- PyTorch
- torchvision
- PIL (Pillow)
- numpy
- scipy
- pandas
- matplotlib
- jupyter (for notebook usage)

### Hardware Recommendations
- **CPU**: Multi-core processor for batch processing
- **GPU**: Optional but recommended for LPIPS and CLIP-IQA metrics
- **Memory**: 8GB+ RAM for large dataset processing
- **Storage**: Sufficient space for image datasets

## 📈 Use Cases

### Research Applications
- **Metric Development**: Compare new IQA metrics against established ones
- **Algorithm Evaluation**: Assess image processing algorithm quality
- **Dataset Analysis**: Analyze quality distributions in image datasets
- **Perceptual Studies**: Correlate computational metrics with human perception

### Practical Applications
- **Image Compression**: Evaluate compression algorithm quality
- **Image Enhancement**: Assess enhancement algorithm effectiveness
- **Quality Control**: Automated image quality assessment in pipelines
- **Benchmarking**: Compare different image processing methods

## 🎓 Getting Started Guide

### For Beginners
1. **Start with basic metric evaluation** using simple image pairs
2. **Explore the Jupyter notebook** [`data.ipynb`](data.ipynb) for interactive analysis
3. **Try different metrics** to understand their characteristics
4. **Load sample datasets** to see batch processing capabilities

### For Researchers
1. **Use the LIVE-MD dataset** for metric validation
2. **Implement custom evaluation pipelines** using the core metrics
3. **Analyze metric correlations** with human perceptual judgments
4. **Extend the system** with new metrics or datasets

### For Developers
1. **Integrate IQA metrics** into existing image processing pipelines
2. **Batch process large image collections** for quality assessment
3. **Build quality-aware applications** using the metric outputs
4. **Optimize processing** for specific use cases and hardware

## 📊 Understanding Metric Outputs

### Full-Reference Metrics (require reference image)
- **Higher SSIM/MS-SSIM** = Better quality (closer to 1.0)
- **Higher PSNR** = Better quality (typically 20-40 dB)
- **Lower LPIPS** = Better quality (closer to 0.0)
- **Higher FSIM/VSI** = Better quality (closer to 1.0)
- **Lower GMSD** = Better quality (closer to 0.0)

### No-Reference Metrics (no reference needed)
- **Lower BRISQUE** = Better quality (0-100 scale)
- **Lower NIQE** = Better quality (lower is better)
- **Higher CLIP-IQA** = Better quality (0-1 scale)

## 🔬 Advanced Features

### Metric Customization
- **Device Selection**: CPU/GPU processing options
- **Image Preprocessing**: Configurable target sizes and normalization
- **Batch Processing**: Efficient evaluation of multiple image pairs
- **Error Handling**: Robust processing with fallback options

### Data Analysis
- **Statistical Analysis**: Correlation analysis between metrics
- **Visualization Tools**: Plot metric distributions and relationships
- **Export Capabilities**: Save results in various formats (CSV, JSON)
- **Interactive Exploration**: Jupyter notebook interface

## 🎯 Best Practices

### Metric Selection
- **Use multiple metrics** for comprehensive assessment
- **Choose appropriate metrics** for your specific use case
- **Consider reference availability** (full-reference vs. no-reference)
- **Validate against human perception** when possible

### Performance Optimization
- **Use GPU acceleration** for deep learning-based metrics
- **Batch process images** for efficiency
- **Optimize image sizes** for your specific requirements
- **Cache results** for repeated evaluations

## 📚 Further Reading

For more detailed usage examples and advanced features, refer to:
- **System Documentation**: Comprehensive guides and examples
- **Research Papers**: Original metric publications and comparisons
- **Code Documentation**: Inline comments and function descriptions
- **Community Resources**: Forums and discussion groups

---

**Note**: This system is designed for research and educational purposes. For production use, ensure proper validation and testing for your specific application domain.