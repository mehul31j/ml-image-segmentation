# Image Segmentation with Scribble Supervision

An advanced image segmentation project featuring both classical K-Nearest Neighbors (KNN) and deep learning U-Net approaches for binary image segmentation. This project compares traditional machine learning methods with modern deep learning architectures, optimized for GPU acceleration using Google Colab T4 runtime.

---

## Project Structure
```text
.
├── challenge.py                   # Basic KNN segmentation pipeline (k=3)
├── challenge_final_gpu.py         # Advanced GPU-optimized training with U-Net comparison
├── unet_model.py                 # U-Net architecture implementations
├── util.py                       # Utility functions (dataset I/O, KNN model, visualization)
├── dataset_utils.py              # Dataset handling and data loading utilities
├── championship_120e_model.pth   # Pre-trained U-Net model (120 epochs)
├── ultimate_0.94_model.pth       # High-performance model (94% accuracy)
├── ultra_model.pth               # Ultra-optimized model
└── dataset/
    ├── training/
    │   ├── images/
    │   ├── scribbles/
    │   ├── ground_truth/
    │   └── predictions/           # Generated predictions
    └── test/
        ├── images/
        ├── scribbles/
        └── predictions/           # Generated test predictions
```

---

## Requirements

### For Local Development
```bash
pip install numpy pillow matplotlib scikit-learn torch torchvision
```

### For Google Colab (Recommended for GPU Training)
This project is optimized to run on Google Colab with T4 GPU for fast training and inference. The GPU-accelerated training provides significant speedup compared to CPU-only execution.

```python
# In Google Colab
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### Basic KNN Approach (k=3)
```bash
python challenge.py
```

### Advanced GPU-Optimized Training with Model Comparison
```bash
python challenge_final_gpu.py
```

This will:
1. Load the training dataset
2. Compare KNN (k=5) vs U-Net performance
3. Train advanced U-Net models with GPU acceleration (100 epochs)
4. Generate high-quality segmentation masks
5. Provide detailed IoU metrics and performance analysis
6. Save optimized models for future use

---

## Project Components

### challenge.py
- Basic segmentation pipeline using KNN (k=3)
- Loads training and test datasets
- Performs baseline segmentation
- Stores predicted masks and visualizes results

### challenge_final_gpu.py
- **Advanced GPU-optimized training pipeline**
- Compares KNN (k=5) baseline with U-Net deep learning approach
- Features 100-epoch training with T4 GPU acceleration
- Includes model comparison and performance metrics
- Optimized for Google Colab environment

### unet_model.py
- Contains multiple U-Net architecture implementations
- Includes SimpleUNet and ImprovedUNet variants
- Designed for efficient GPU training and inference

### util.py
A comprehensive utility module containing:
- **Dataset Handling**: load_dataset(), store_predictions()
- **KNN Model**: segment_with_knn() with configurable k values
- **Visualization**: Advanced plotting and result visualization
- **Evaluation**: evaluate_binary_miou(), IoU metrics calculation

### dataset_utils.py
- Advanced data loading and preprocessing utilities
- Batch processing and GPU data transfer optimization

---


## Model Performance

This project demonstrates a comprehensive comparison between traditional machine learning and deep learning approaches:

- **KNN Baseline (k=5)**: Fast, interpretable classical approach
- **U-Net Architecture**: State-of-the-art deep learning segmentation
- **GPU Acceleration**: Optimized training on Google Colab T4 GPU
- **Performance Metrics**: Detailed IoU analysis and model comparison

The U-Net model significantly outperforms the KNN baseline, achieving superior segmentation quality through deep feature learning and spatial context understanding.

---

## GPU Training on Google Colab

This project is specifically optimized for Google Colab's T4 GPU runtime:

1. **Upload the project** to Google Colab or mount from Google Drive
2. **Enable GPU runtime**: Runtime → Change runtime type → GPU (T4)
3. **Run the GPU-optimized script**: `challenge_final_gpu.py`
4. **Monitor training progress** with real-time loss and IoU metrics
5. **Download trained models** for future inference

## Scribble Supervision Format

- **Scribble masks**:
  - `0`: Background pixels
  - `1`: Foreground pixels  
  - `255`: Unlabeled/unknown pixels

- **Predictions and ground truth**: Binary masks using `0` (background) and `1` (foreground)

---

## Features

- 🔬 **Dual Approach**: Compare KNN vs U-Net segmentation methods
- 🚀 **GPU Accelerated**: Optimized for Google Colab T4 runtime
- 📊 **Performance Analysis**: Comprehensive IoU metrics and visualization
- 💾 **Model Persistence**: Save and load trained models
- 🎯 **High Accuracy**: Achieve 94%+ segmentation accuracy with optimized models
- 📈 **Real-time Monitoring**: Track training progress with detailed metrics

---

## Contributing

This project demonstrates advanced image segmentation techniques. Feel free to:
- Experiment with different U-Net architectures
- Optimize hyperparameters for better performance  
- Extend to multi-class segmentation scenarios
- Implement additional evaluation metrics

---

## License

Open source project - feel free to use and modify for your research and projects.

---

**Happy Segmenting! 🎯**
