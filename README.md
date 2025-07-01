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
- **Inference speed**: ~10-50ms per frame (depending on hardware)
- **Accuracy**: Varies by object class and fusion method
- **Memory usage**: ~100-200MB for model and buffers

### Data Generation
- **Scenarios per run**: Configurable (typically 10-50)
- **Frames per scenario**: Variable based on waypoints
- **Total data size**: ~1-10GB per complete run

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

## 📞 Support

For questions and support:
- Check the `Presentation/` folder for detailed documentation
- Review the code comments for implementation details
- Open an issue for bugs or feature requests

---

**Note**: This project requires Webots simulation environment and proper Python dependencies. Ensure all paths are correctly configured for your system setup.
