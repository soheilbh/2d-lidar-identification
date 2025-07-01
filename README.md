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

![Training Process](training.gif)

**[Full Training Video](training.mp4)** – Watch the data generation process with 3 main scenarios and 5 random predefined robot positions, demonstrating how training data is collected for the RGB encoding approach.

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

### 📝 Note on Extra Scripts
The `scripts/Extra scripts/` folder contains additional utility scripts that may require path adjustments:
- **Usage**: Move scripts to the main `scripts/` folder for correct path resolution
- **Alternative**: Modify input/output paths in the scripts to match your directory structure
- **Purpose**: These scripts provide additional analysis, testing, and processing capabilities

## 🤖 Main Controllers

### 1. `full_train_multi_scenario_position_controller.py`
**Purpose**: Primary data generation controller for creating training datasets
- **Functionality**:
  - Executes predefined scenarios with different object configurations
  - Collects raw LiDAR data with object annotations
  - Generates comprehensive training datasets
  - Supports multiple robot positions and waypoints
- **Output**: Raw LiDAR logs in JSONL format
- **Usage**: Main controller for dataset generation

### 2. `robot_controller_test_model.py`
**Purpose**: Model testing and real-time inference controller
- **Functionality**:
  - Loads trained YOLOv8 models
  - Performs real-time object detection
  - Provides live visualization of detections
  - Supports keyboard-controlled robot movement
- **Features**:
  - Live RGB LiDAR visualization
  - Real-time YOLO inference
  - World coordinate conversion
  - Performance metrics display

## 🔧 Core Scripts

### Data Generation Scripts
- **`multi_frame_scenarios_simple_fused_rgb_generator.py`**: Creates RGB images from simple fused LiDAR data (3-frame concatenation)
- **`multi_frame_scenarios_aligned_fused_rgb_generator.py`**: Creates RGB images from aligned fused LiDAR data (5-frame pose-based fusion)
- **`multi_frame_scenarios_simple_fused_label_generator.py`**: Generates YOLO labels for simple fused data
- **`multi_frame_scenarios_aligned_fused_label_generator.py`**: Generates YOLO labels for aligned fused data

### Processing Scripts
- **`post_process.py`**: Post-processes LiDAR data for object detection
- **`split_dataset.py`**: Splits datasets into training/validation sets
- **`inference.py`**: Standalone inference script for model testing

### Visualization Scripts
- **`visualize_labels_simple_fused.py`**: Visualizes simple fused data with bounding boxes
- **`visualize_labels_aligned_fused.py`**: Visualizes aligned fused data with bounding boxes

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Data Generation
```bash
# Run the main data generation controller in Webots
# This will generate raw LiDAR logs in raw_logs/
```

### 2. Data Processing
```bash
# Generate multi-frame fused data
python scripts/multi_frame_scenarios_simple_fused_rgb_generator.py
python scripts/multi_frame_scenarios_simple_fused_label_generator.py

# Or for aligned fusion
python scripts/multi_frame_scenarios_aligned_fused_rgb_generator.py
python scripts/multi_frame_scenarios_aligned_fused_label_generator.py
```

### 3. Model Training
```bash
# Train YOLOv8 model on processed data
yolo train model=yolov8n.pt data=path/to/data.yaml epochs=120
```

### 4. Model Testing
```bash
# Test model in Webots simulation
# Use robot_controller_test_model.py for real-time inference
```

## 📊 Data Formats

### Raw LiDAR Data
- **Format**: JSONL (JSON Lines)
- **Content**: Frame ID, raw scan, robot pose, object details, scenario information
- **Location**: `raw_logs/`

### Processed Data
- **RGB Images**: 64x384 PNG files (3-channel fused LiDAR data)
- **Labels**: YOLO format text files
- **Location**: `output/multi_frame/multi_scenarios/`

### Model Outputs
- **Trained Models**: YOLOv8 weights and ONNX/TorchScript exports
- **Training Results**: Metrics, confusion matrices, validation plots
- **Location**: `training_outputs/`

## 🎮 Usage Examples

### Generate Training Data
1. Open Webots and load `bot3/worlds/bot3.wbt`
2. Run `full_train_multi_scenario_position_controller.py`
3. Wait for all scenarios to complete
4. Check `raw_logs/` for generated data

### Test Trained Model
1. Ensure trained model is in `training_outputs/`
2. Open Webots and load the world
3. Run `robot_controller_test_model.py`
4. Use keyboard controls to move robot and observe detections

### Visualize Results
```bash
# Visualize simple fused data
python scripts/visualize_labels_simple_fused.py

# Visualize aligned fused data
python scripts/visualize_labels_aligned_fused.py
```

## 🔧 Configuration

### Global Configuration (`bot3/controllers/global_config.py`)
- **Movement parameters**: Speed, turning gains, waypoint thresholds
- **Safety settings**: Obstacle avoidance, safety margins
- **Simulation settings**: Time steps, scan distances

### Model Configuration
- **Model path**: Update in `robot_controller_test_model.py`
- **Inference settings**: Confidence thresholds, NMS parameters
- **Buffer settings**: Frame buffer size for multi-frame fusion

## 📈 Performance

### Model Performance
- **Inference speed**: ~6-48ms per frame (depending on hardware)
- **Accuracy**: 98.4% mAP@0.5 on test set
- **Memory usage**: ~100-200MB for model and buffers
- **Real-time capability**: 21 FPS on Raspberry Pi 5

### Data Generation
- **Scenarios per run**: 160 unique scenarios
- **Positions per scenario**: 90 random robot positions
- **Total samples**: 768,897 labeled RGB tensors
- **Training time**: ~18 hours on H100 GPU

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with Webots simulation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Webots**: Robot simulation environment
- **YOLOv8**: Object detection framework
- **TurtleBot3**: Robot platform
- **Ultralytics**: YOLO implementation

## 📞 Support & Resources

- For presentations, documentation, and training demos, see the [`resource/`](resource/) folder.
- For technical details, code explanations, and implementation notes, review the code comments and the [`README.md`](README.md).
- For additional resources, training curves, and result images, check the [`training_outputs/`](training_outputs/) and [`output/`](output/) folders.
- If you have questions, bug reports, or feature requests, please [open an issue](https://github.com/soheilbh/2d-lidar-identification/issues) on GitHub.

---

**Note**: This project requires Webots simulation environment and proper Python dependencies. Ensure all paths are correctly configured for your system setup.
