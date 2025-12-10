# ADMN-RealWorld

**Adaptive Multimodal Deep Network for Real-World RGB-D Gesture Recognition**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 📄 **Full Project Report**: [https://alanhsiu.github.io/ECM202A_2025Fall_Project_14/](https://alanhsiu.github.io/ECM202A_2025Fall_Project_14/)

---

## 📖 Overview

This project implements an **Adaptive Multimodal Deep Network (ADMN)** for robust gesture recognition using RGB-D data. The system dynamically allocates computational resources across RGB and Depth modalities based on input quality, achieving **100% accuracy** with the 12-layer adaptive budget while using only half the layers of a fixed 24-layer baseline.

### Key Features

- 🎯 **Adaptive Layer Allocation**: Dynamically adjusts layer usage based on input quality
- 🌈 **Multi-Modal Fusion**: Combines RGB and Depth for robust recognition
- 📊 **Real-World Robustness**: Handles corrupted inputs (occlusions, low light)
- ⚡ **Edge Deployment**: Deployed on Raspberry Pi 5 for real-time inference

---

## 👥 Team

| Name | Role | GitHub |
|------|------|--------|
| **Cheng-Hsiu (Alan) Hsieh** | Project Lead / ML Engineer | [@Alanhsiu](https://github.com/Alanhsiu) |
| **Daniel Lee** | Hardware Integration | [@Daniel-Lee-1106](https://github.com/Daniel-Lee-1106) |
| **Ting-Yu Yeh** | Hardware Integration | [@TingYu0225](https://github.com/TingYu0225) |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Alanhsiu/ADMN-RealWorld.git
cd ADMN-RealWorld

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Structure

```
data/
├── clean/
│   ├── standing/
│   │   ├── color_image_0.png
│   │   └── depth_image_0.png
│   ├── left_hand/
│   ├── right_hand/
│   └── both_hands/
├── depth_occluded/
│   └── [same structure]
└── low_light/
    └── [same structure]
```

- Full dataset download (for `data_new/`): [Google Drive folder](https://drive.google.com/drive/folders/17sohVmte4j93pvPY2eXT6pf6A9uESkiA?usp=sharing) containing `clean`, `depth_occluded`, `low_light`.
- Place the downloaded folders inside `data_new/` (i.e., `data_new/clean`, `data_new/depth_occluded`, `data_new/low_light`).

---

## 🎓 Training

### Quick Run (Stage 1 → Stage 2)

```bash
bash software/run.sh
```

### Stage 1: Baseline Classifier

```bash
python software/scripts/train_stage1.py \
    --data_dir data \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --output_dir checkpoints/stage1
```

### Stage 2: Adaptive Controller

```bash
python software/scripts/train_stage2.py \
    --stage1_checkpoint checkpoints/stage1/best_model.pth \
    --data_dir data \
    --total_layers 12 \
    --output_dir checkpoints/stage2
```

### Run Baselines

```bash
bash software/run_baselines.sh
```

### Reproduce Reported Results

1) Download the full dataset from the Google Drive link above and place it under `data_new/` (preserves the `clean/depth_occluded/low_light` subfolders).  
2) Train and evaluate:  
   - Quick pipeline: `bash software/run.sh` (Stage 1 + Stage 2).  
   - Baseline suite: `bash software/run_baselines.sh` (includes dynamic/naive/reduced budgets).  
3) Results and logs will appear in `checkpoints/`, `logs/`, and `results/baselines/` as in the report.

---

## 🧪 Inference

### Single-Sample Inference

```python
import sys

# Make project modules available when running from repo root
sys.path.append("software")

import torch
from PIL import Image
from models.adaptive_controller import AdaptiveGestureClassifier
from data.gesture_dataset import rgb_transform, depth_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['standing', 'left_hand', 'right_hand', 'both_hands']

# Load model
model = AdaptiveGestureClassifier(
    num_classes=4,
    total_layers=12,
    stage1_checkpoint='checkpoints/stage1/best_model.pth'
).to(device)
ckpt = torch.load('checkpoints/stage2/best_controller_12layers.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Inference
rgb = rgb_transform(Image.open('test_rgb.png')).unsqueeze(0).to(device)
depth = depth_transform(Image.open('test_depth.png')).unsqueeze(0).to(device)

with torch.no_grad():
    logits, allocation = model(rgb, depth, temperature=0.5, return_allocation=True)
    pred = logits.argmax(dim=1).item()

print(f"Prediction: {classes[pred]}")
print(f"RGB layers: {allocation[0, 0].sum().item():.0f}/12")
print(f"Depth layers: {allocation[0, 1].sum().item():.0f}/12")
```

---

## 🤖 Raspberry Pi 5 Deployment

### Setup

```bash
sudo apt update
sudo apt upgrade -y

# Realsense and pyrealsense set up for Raspberry Pi 5

## Install packages
sudo apt-get install -y libdrm-amdgpu1 libdrm-amdgpu1-dbgsym libdrm-dev libdrm-exynos1 libdrm-exynos1-dbgsym libdrm-freedreno1 libdrm-freedreno1-dbgsym libdrm-nouveau2 libdrm-nouveau2-dbgsym libdrm-omap1 libdrm-omap1-dbgsym libdrm-radeon1 libdrm-radeon1-dbgsym libdrm-tegra0 libdrm-tegra0-dbgsym libdrm2 libdrm2-dbgsym

sudo apt-get install -y libglu1-mesa libglu1-mesa-dev glusterfs-common libglu1-mesa libglu1-mesa-dev libglui-dev libglui2c2

sudo apt-get install -y libglu1-mesa libglu1-mesa-dev mesa-utils mesa-utils-extra xorg-dev libgtk-3-dev libusb-1.0-0-dev

## Download librealsense v2.50.0
https://github.com/realsenseai/librealsense/archive/refs/tags/v2.50.0.zip

unzip v2.50.0.zip
cd librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/ 
sudo udevadm control --reload-rules && sudo udevadm trigger

## Install protobuf
cd ~
git clone --depth=1 -b v3.5.1 https://github.com/google/protobuf.git
cd protobuf
./autogen.sh
./configure
make -j$(nproc)
sudo make install
cd python
export LD_LIBRARY_PATH=../src/.libs
python3 setup.py build --cpp_implementation 
python3 setup.py test --cpp_implementation
sudo python3 setup.py install --cpp_implementation
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3
sudo ldconfig
protoc --version

## Install TBB
cd ~
wget https://github.com/PINTO0309/TBBonARMv7/raw/master/libtbb-dev_2018U2_armhf.deb
sudo dpkg -i ~/libtbb-dev_2018U2_armhf.deb
sudo ldconfig
rm libtbb-dev_2018U2_armhf.deb

## Install OpenCV (C++ Library for RealSense)
wget https://github.com/mt08xx/files/raw/master/opencv-rpi/libopencv3_3.4.3-20180907.1_armhf.deb
sudo apt install -y ./libopencv3_3.4.3-20180907.1_armhf.deb
sudo ldconfig

## Install RealSense SDK/librealsense
cd ~/librealsense-2.50.0
mkdir  build  && cd build
cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=Release -DFORCE_LIBUVC=true
make -j$(nproc)
sudo make install

## Install pyrealsense2
cd ~/librealsense-2.50.0/build
cmake .. -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=$(which python3)
make -j$(nproc)
sudo make install

### Add python path
vim ~/.zshrc
export PYTHONPATH=$PYTHONPATH:/usr/local/lib

source ~/.zshrc

## Try RealSense
realsense-viewer

# Libraries Installation for software function

## Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

## Install OpenCv for python
pip install opencv-python

## Install gpiozero for GPIO control
pip install gpiozero

## Install pyaudio for enabling microphone audio
pip install pyaudio

## Install psutil for providing system information
pip install psutil
```

### Real-Time Inference

```bash
python3 software/scripts/realtime_inference.py \
    --model_path checkpoints/stage2/best_controller_12layers.pth \
    --camera_id 0
```

### Performance on Raspberry Pi 5

| Total Layers | GFLOPs | Latency (ms) | Accuracy |
|--------------|--------|--------------|----------|
| 8 | 3.97 | 521 | 98.33% |
| 12 | 5.84 | 727 | 100.00% |
| 24 (baseline) | 11.43 | 1201 | 100.00% |

---

## 📊 Results Summary

| Model | Layers | Accuracy |
|-------|--------|----------|
| Stage 1 (Upper Bound) | 24 | 100.00% |
| Stage 2 Adaptive | 12 | 100.00% |
| Stage 2 Adaptive | 8 | 98.33% |
| Stage 2 Adaptive | 6 | 80.00% |
| Stage 2 Adaptive | 4 | 37.50% |

### Learned Allocation Patterns

| Corruption | RGB Layers | Depth Layers |
|------------|------------|--------------|
| Clean | 6.1 | 5.9 |
| Depth Occluded | 7.5 | 4.5 |
| Low Light | 2.0 | 10.0 |

> For detailed results and analysis, see the [full report](https://alanhsiu.github.io/ECM202A_2025Fall_Project_14/).

---

## 📁 Project Structure

```
ADMN-RealWorld/
├── software/                # Code & run scripts
│   ├── run.sh               # Quick train script
│   ├── run_baselines.sh     # Baseline experiments
│   ├── scripts/             # Training & inference
│   │   ├── train_stage1.py
│   │   ├── train_stage2.py
│   │   ├── inference_stage1.py
│   │   └── inference_stage2.py
│   ├── models/              # Model architectures
│   │   ├── gesture_classifier.py
│   │   └── adaptive_controller.py
│   ├── GTDM_Lowlight/       # ViT backbone and components
│   │   └── models/
│   ├── rpi/                 # Raspberry Pi inference helpers
│   └── utils/               # Utilities & visualization
├── data/                    # Dataset & loaders
├── checkpoints/             # Saved models
├── results/                 # Metrics & visualizations
├── doc/                     # Project website
└── requirements.txt
```

---

## 🛠️ Development

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug"
git commit -m "docs: update documentation"
```

### Branching

```bash
git checkout -b feature/your-feature-name
# Make changes, then PR to main
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [ADMN Paper](https://arxiv.org/html/2502.07862v2) - Original framework
- [NESL Lab](http://nesl.ee.ucla.edu/) - Thanks to TA **Jason Wu** for guidance

---

## 📧 Contact

**Alan Hsieh** - [alanhsiu@ucla.edu](mailto:alanhsiu@ucla.edu)

---

<div align="center">

[📄 Full Report](https://alanhsiu.github.io/ECM202A_2025Fall_Project_14/) | [🐛 Report Bug](https://github.com/Alanhsiu/ECM202A_2025Fall_Project_14/issues)

</div>
