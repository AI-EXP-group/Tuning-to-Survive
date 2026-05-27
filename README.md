# Tuning to Survive: Neural Network Watermarking Defense Framework

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Project Overview

This project implements the **"Tuning to Survive"** neural network watermarking defense method, designed to protect the intellectual property of deep learning models. The framework systematically investigates the threat of Model Extraction Attacks on neural network watermarks and proposes an effective defense strategy.

### Paper Information

- **Paper Link**: [FINE-TUNING MODEL WATERMARKS AGAINST EXTRACTION ATTACKS BY REHEARSAL](https://ieeexplore.ieee.org/abstract/document/11463710)

## 🎯 Core Features

### Defense Phase
1. **Train Clean Model** (`train_clean.py`): Train the base model without embedded watermarks
2. **Generate Trigger** (`get_trigger.py`): Generate trigger patterns for watermark embedding
3. **Embed Watermark** (`watermarking.py`): Embed ownership watermarks into the model
4. **Tuning to Survive** (`tuning.py`): Core defense method to enhance watermark robustness against model extraction attacks

### Attack Phase (Attack Testing)
1. **Model Extraction Attack** (`extraction.py`):
   - Soft Label Extraction
   - Hard Label Extraction
   - Different surrogate dataset attacks
   - Different stolen model architecture attacks

2. **Pruning Attack** (`pruning.py`): Test watermark robustness against model pruning
3. **Quantization Attack** (`quantization.py`): Test watermark robustness against model quantization

## 🏗️ Project Structure

```
tuning-to-survive/
├── checkpoint/              # Model checkpoint save directory
│   └── cifar10/
│       └── res18/
│           ├── clean/       # Clean models
│           ├── watermarked/ # Watermarked models
│           └── t2s/         # Models after T2S defense
├── networks/                # Neural network architecture definitions
│   ├── resnet.py           # ResNet series
│   ├── wresnet.py          # Wide ResNet
│   ├── densenet.py         # DenseNet
│   ├── googlenet.py        # GoogleNet
│   ├── mobilenetv2.py      # MobileNetV2
│   ├── mobilevit.py        # MobileViT
│   ├── vit.py              # Vision Transformer
│   └── ...
├── functional_model/        # Functional model interfaces
├── scripts/                 # Core scripts
│   ├── config.yaml         # Experiment configuration file
│   ├── run_pipeline.sh     # Unified run script
│   ├── defense/            # Defense-related scripts
│   ├── attack/             # Attack testing scripts
│   └── validation/         # Validation scripts
└── utils/                   # Utility functions
    ├── utils.py
    └── watermark_utils.py
```

## 🚀 Quick Start

### Configure Experiment Parameters

Edit the `scripts/config.yaml` file:

```yaml
# Basic configuration
device: "cuda:0"              # GPU device
data_path: "/path/to/data"    # Dataset path
dataset: "cifar10"            # Dataset: cifar10 | cifar100 | tinyimagenet
model: "res18"                # Model: res18 | wrn | dense121 | googlenet | ...

# Watermark label configuration
source_label1: 9              # Source class label
target_label: 6               # Target class label (watermark label)
```

### Run Complete Pipeline

```bash
cd scripts
bash run_pipeline.sh
```

Or specify a configuration file:

```bash
bash run_pipeline.sh path/to/config.yaml
```

## 🔬 Experiment Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    Defense Phase                        │
├─────────────────────────────────────────────────────────┤
│  1. Train Clean Model                                   │
│     ↓                                                   │
│  2. Generate Trigger                                    │
│     ↓                                                   │
│  3. Embed Watermark                                     │
│     ↓                                                   │
│  4. Tuning to Survive (Defense)                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    Attack Phase                         │
├─────────────────────────────────────────────────────────┤
│  5. Model Extraction Attack                             │
│     - Soft/Hard Label Extraction                        │
│     - Different Surrogate Datasets                      │
│     - Different Stolen Model Architectures              │
│     - Double Extraction                                 │
│     ↓                                                   │
│  6. Pruning Attack                                      │
│     ↓                                                   │
│  7. Quantization Attack                                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   Validation Phase                      │
├─────────────────────────────────────────────────────────┤
│  8. Test Model Accuracy                                 │
│  9. Verify Watermark Success Rate                       │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Advanced Usage

### Run Individual Stages

```python
# Train clean model
python scripts/defense/train_clean.py \
    --idx 0 --dataset cifar10 --model res18 \
    --epochs 30 --lr 0.1 --batch_size 256

# Embed watermark
python scripts/defense/watermarking.py \
    --idx 0 --dataset cifar10 --model res18 \
    --mode feature --epochs 5 --num 500

# Run T2S defense
python scripts/defense/tuning.py \
    --idx 0 --dataset cifar10 --model res18 \
    --mode feature --alpha 50

# Model extraction attack
python scripts/attack/extraction.py \
    --idx 0 --target_dataset cifar10 --target_model res18 \
    --stolen_model res18 --sur_dataset cifar10
```

## 📝 Citation

If you use the code or methods from this project, please cite:

```bibtex
@INPROCEEDINGS{11463710,
  author={Zhang, Weibin and Mei, Jian-Ping and Yu, Miaoqi and Zhu, Tiantian and Xiao, Jie},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Fine-Tuning Model Watermarks Against Extraction Attacks by Rehearsal}, 
  year={2026},
  pages={13967-13971},
  keywords={Feedback;Circuits;Protocols;HTTP;Radio access networks;Regional area networks;Learning (artificial intelligence);Artificial neural networks;Artificial intelligence;Neural networks;model watermarking;model stealing;model intellectual property;stealing simulation},
  doi={10.1109/ICASSP55912.2026.11463710}}
```
