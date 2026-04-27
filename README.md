# Deepfake Detection - FYP Project

Deep learning based deepfake video detection using EfficientNet-B0.

## 🎯 Important Files (Keep in Git)

### Core Scripts
- `preprocessing.py` - Preprocesses videos into frames (32 quarters system)
- `train_incremental.py` - Trains model incrementally on CPU
- `test_model.py` - Tests trained model on videos with face detection
- `requirements.txt` - Python dependencies

### Final Model
- `model_output/final_model.pth` - Trained model (100% validation accuracy)

### Helper Scripts
- `run_all_quarters.sh` - Automate preprocessing
- `run_training.sh` - Automate training
- `combine_quarters.py` - Utility (not used in current workflow)

## 🚫 Ignored Files (.gitignore)

### Large Folders (~11.6 GB total)
- `venv/` - Virtual environment (1.7 GB)
- `preprocessed_data_pytorch/` - Preprocessed frames (8.0 GB)
- `model_output/model_after_quarter_*.pth` - 32 intermediate checkpoints (1.8 GB)

### Test Files
- `test_video.mp4`, `test-1.mp4` - Test videos
- `test_results/` - Test output folders

### Generated Files
- Training plots (confusion_matrix.png, roc_curve.png, etc.)
- Downloaded models (haarcascade_frontalface_default.xml)
- Jupyter notebooks (.ipynb files)
- Python cache (`__pycache__/`)

## 📦 Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Preprocess Dataset
```bash
python preprocessing.py --quarter 1  # Process quarter 1-32
```

### 2. Train Model
```bash
python train_incremental.py
```

### 3. Test Model
```bash
python test_model.py
```

## 🏗️ Model Architecture

- **Backbone**: EfficientNet-B0 (pretrained)
- **Classifier**: 1280 → 512 → 256 → 1 (Binary)
- **Training**: 32 quarters, 3 epochs each
- **Accuracy**: 100% validation accuracy

## 📊 Dataset

- **Source**: FaceForensics++_C23
- **Size**: ~1,728 videos processed (27 per quarter × 32 quarters)
- **Frames**: ~12,000 frames total
- **Split**: 70% train, 15% val, 15% test

## 🎨 Features

- Frame-by-frame analysis
- Face detection and highlighting
- Detailed reports (TXT, JSON, images)
- Summary grid visualization
- CPU-optimized processing
