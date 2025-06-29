# Quick Start Guide - YOLOv11-CFruit

## 📋 Overview

This guide will help you quickly train YOLOv11-CFruit models using fruit data annotated with labelme format.

---

**Last Updated: June 2024**  
**Document Version: v1.0**

---

## 🚀 Environment Setup

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or use conda
conda install pytorch torchvision torchaudio -c pytorch
pip install -r requirements.txt
```

### 2. Check Environment

```bash
python test_project.py
```

## 📊 Data Preparation

### Data Format Requirements

Your data should contain:
- Image files: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` format
- Annotation files: labelme format `.json` files

### Directory Structure Example

```
your_data/
├── image1.jpg
├── image1.json
├── image2.jpg
├── image2.json
└── ...
```

### Quick Data Preparation

```bash
# Use sample data for demonstration
python examples/prepare_and_train.py --create-sample

# Use real data (supports circular annotations)
python scripts/prepare_data_circle_fixed.py --input-dir /path/to/your/data --output-dir data/cfruit --class-names cfruit
```

## 🎯 Model Training

### One-Click Training (Recommended)

```bash
# Use automatic training script
./scripts/auto_train_and_visualize.sh

# Use quick test script
./scripts/quick_auto_train.sh
```

### Step-by-Step Training

#### Step 1: Data Preparation

```bash
python scripts/prepare_data_circle_fixed.py \
    --input-dir /path/to/your/data \
    --output-dir data/cfruit \
    --class-names cfruit
```

#### Step 2: Start Training

```bash
# Improved training (recommended)
python scripts/train_improved.py \
    --config configs/model/yolov11_cfruit_improved.yaml \
    --data configs/data/cfruit.yaml \
    --epochs 100 \
    --batch-size 16 \
    --save-dir checkpoints

# Simple training
python scripts/simple_train.py \
    --config configs/model/yolov11_cfruit.yaml \
    --data configs/data/cfruit.yaml \
    --epochs 100 \
    --batch-size 16 \
    --save-dir checkpoints
```

## ⚙️ Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | Required | Directory containing images and JSON files |
| `--class-names` | cfruit | List of class names |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 16 | Batch size |
| `--img-size` | 640 | Input image size |
| `--save-dir` | checkpoints | Model save directory |
| `--device` | auto | Training device (cuda/cpu/auto) |

## 📈 Monitor Training

### TensorBoard

```bash
tensorboard --logdir logs
```

Visit http://localhost:6006 to view training curves.

### Training Logs

Training logs are saved in `logs/train.log`, including:
- Loss value changes
- Learning rate changes
- Validation metrics
- Model save information

## 🧪 Model Testing

After training, you can test the model:

```bash
python examples/basic_detection.py \
    --model checkpoints/best.pt \
    --image /path/to/test/image.jpg
```

## ❓ FAQ

### Q: How to handle images of different sizes?
A: Training will automatically resize to uniform size (default 640x640) while maintaining aspect ratio.

### Q: What if annotation quality is poor?
A: Suggestions:
1. Check annotation accuracy
2. Increase data augmentation
3. Adjust learning rate
4. Increase training epochs

### Q: Training is slow, what to do?
A: You can:
1. Reduce batch size
2. Use GPU training
3. Reduce image size
4. Use pre-trained weights

### Q: How to add new classes?
A: Modify the `--class-names` parameter, for example:
```bash
--class-names cfruit unripe_cfruit
```

## 🚀 Performance Optimization Tips

### Data Quality
- Ensure annotation accuracy
- Balance sample numbers across classes
- Increase data diversity

### Training Strategy
- Use pre-trained weights
- Adjust learning rate scheduling
- Use mixed precision training
- Enable EMA (Exponential Moving Average)

### Hardware Optimization
- Use GPU training
- Adjust batch size
- Optimize data loading
- Use SSD storage

## 📝 Complete Example

```bash
# 1. Create sample data and train
python examples/prepare_and_train.py --create-sample --epochs 10

# 2. Train with real data
python scripts/prepare_data_circle_fixed.py \
    --input-dir /path/to/your/data \
    --output-dir data/cfruit \
    --class-names cfruit

python scripts/train_improved.py \
    --config configs/model/yolov11_cfruit_improved.yaml \
    --data configs/data/cfruit.yaml \
    --epochs 100 \
    --batch-size 16 \
    --save-dir checkpoints

# 3. Monitor training
tensorboard --logdir logs

# 4. Test model
python examples/basic_detection.py \
    --model checkpoints/best.pt \
    --image /path/to/test/image.jpg
```

## 🔗 Next Steps

After training, you can:
1. Test model performance
2. Deploy to production
3. Optimize model architecture
4. Collect more data

For detailed usage instructions, please refer to [docs/data_preparation.md](docs/data_preparation.md) and [USAGE.md](USAGE.md). 