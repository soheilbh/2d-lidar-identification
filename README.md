# 2D LiDAR Object Identification

A comprehensive project for object identification and classification using 2D LiDAR data on TurtleBot3. This project implements end-to-end object detection using YOLOv8 models trained on fused multi-frame LiDAR data.

## 🎯 Project Overview

The main goal of this project is to develop a robust object identification system using 2D LiDAR sensors. The system processes raw LiDAR scan data, fuses multiple frames for better object detection, and uses deep learning models to classify objects in real-time.

### Key Features
- **Multi-frame LiDAR fusion**: Combines multiple LiDAR scans for improved object detection
- **Two fusion approaches**: Simple fusion (concatenation) and aligned fusion (pose-based)
- **YOLOv8 integration**: Real-time object detection using trained neural networks
- **Scenario generation**: Automated generation of diverse training scenarios
- **Comprehensive visualization**: Tools for visualizing detections and training data
- **Webots simulation**: Full simulation environment with TurtleBot3

## 🎬 Demo Video

Watch the full simulation in action:

[![2D LiDAR Object Identification Demo](https://img.youtube.com/vi/sjO1z04g8Jg/0.jpg)](https://www.youtube.com/watch?v=sjO1z04g8Jg)

**[Full Simulation Video](https://www.youtube.com/watch?v=sjO1z04g8Jg)** - See the complete system in action with real-time object detection and multi-frame LiDAR fusion.

### Training Example

![Training Process](resource/training.gif)

Data generation process with 3 main scenarios and 5 random predefined robot positions, demonstrating how training data is collected for the RGB encoding approach.

## 🏆 Performance Results

### Method B: Deep Learning Approach (RGB Encoding)

Our deep learning approach achieves state-of-the-art performance on 2D LiDAR object detection:

#### **Overall Performance Metrics:**
- **Mean Average Precision (mAP@0.5)**: 0.984 (98.4%)
- **Mean Average Precision (mAP@0.5:0.95)**: 0.778 (77.8%)
- **Overall Precision**: 94.9%
- **Overall Recall**: 94.7%
- **F1-Score**: 0.95

#### **Per-Class Performance:**

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| Chair | 98.6% | 91.3% | 98.1% | 74.6% |
| Box | 95.7% | 99.6% | 99.3% | 78.0% |
| Desk | 92.2% | 95.4% | 98.2% | 82.2% |
| Doorframe | 93.0% | 92.6% | 98.0% | 76.3% |

#### **Real-time Performance:**
- **Raspberry Pi 5**: 47.8ms per frame (~21 FPS)
- **MacBook Air M2**: 6.2ms per frame (~161 FPS)
- **Raspberry Pi 3**: ~2 seconds per frame (TorchScript mode)

### **Methodology: RGB Encoding of LiDAR Scans**

Our innovative approach encodes consecutive LiDAR scans as compact RGB images:

#### **Data Representation:**
- **Input Format**: 64×384×3 RGB tensor
- **Temporal Encoding**: 3 consecutive LiDAR frames stacked as R, G, B channels
- **Frame Buffer**: Simple FIFO mechanism without pose alignment
- **Spatial Resolution**: 64 distance bins × 360 angular bins (padded to 384)

#### **Training Strategy:**
- **Dataset Size**: 768,897 labeled RGB tensor inputs
- **Training Samples**: 629,591
- **Validation Samples**: 72,528
- **Test Samples**: 38,321 (completely unseen scenarios)
- **Training Duration**: 120 epochs
- **Hardware**: NVIDIA H100 GPU (80GB VRAM)
- **Batch Size**: 1024

#### **Key Innovations:**
- **94% reduction** in input size compared to occupancy grid approaches
- **Direct LiDAR encoding** without intermediate representations
- **Temporal motion capture** through RGB channel stacking
- **Privacy-preserving** camera-free perception
- **Real-time embedded deployment** on low-power hardware

## 📁 Project Structure

```
2d-lidar-identification/
├── bot3/                          # Webots simulation environment
│   ├── controllers/               # Robot controllers
│   │   ├── full_train_multi_scenario_position_controller/  # Main data generation controller
│   │   ├── robot_controller_test_model/                    # Model testing and inference controller
│   │   ├── Extra_controllers/     # Additional experimental controllers
│   │   └── global_config.py      # Global configuration parameters
│   ├── worlds/                    # Webots world files
│   ├── protos/                    # 3D object prototypes
│   └── utils/                     # Utility functions
├── scripts/                       # Main processing scripts
│   ├── Extra scripts/             # Additional utility scripts (see note below)
│   ├── multi_frame_scenarios_*.py # Multi-frame data generation scripts
│   ├── inference.py               # Model inference and testing
│   ├── post_process.py            # LiDAR data post-processing
│   ├── split_dataset.py           # Dataset splitting utilities
│   └── visualize_labels_*.py      # Visualization tools
├── raw_logs/                      # Raw LiDAR data logs
├── output/                        # Processed data and results
├── training_outputs/              # YOLOv8 training results and models
├── Presentation/                  # Project documentation and videos
└── requirements.txt               # Python dependencies
```