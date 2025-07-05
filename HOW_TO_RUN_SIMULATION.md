# How to Run the Full Simulation and Data Pipeline

This guide explains how to generate training data, run the simulation, process the data, and prepare it for YOLO training using this repository.

---

## Pre-generated Data (Google Drive)

For convenience, we provide pre-generated datasets that you can download directly:

### **Google Drive Structure:**
```
📁 Google Drive Folder/
├── 📁 raw_logs/
│   └── yolo1d_scan_2025-06-06_01-11-05.jsonl.tar.gz
├── 📁 Train_and_Test_Splited_Data_simpled_fused/
│   ├── splits_simple_fused_2025-06-06_01-11-05_images_trains.tar.gz
│   ├── splits_simple_fused_2025-06-06_01-11-05_lables_trains.tar.gz
│   └── splits_simple_fused_2025-06-06_01-11-05_val_test.tar.gz
├── 📁 Train_and_Test_Splited_Data_aligned_fused/
│   └── ... (similar structure for aligned fused, if present)
└── 📁 senarios/
    └── 14400_scenarios_20250606_004922.jsonl
```

**Note:**  
- The simple fused split data and raw logs are provided as compressed `.tar.gz` archives.  
- Extract these files to obtain the `train`, `val`, and `test` folders for images and labels, or the raw log file for use in the pipeline.

### **Download Link:**
[Google Drive - 2D LiDAR Identification Data](https://drive.google.com/drive/folders/1stHaVEMeqq-YLNqXO39XQT179cmok1Rd?usp=sharing)

### **Usage:**
- **Skip steps 1-5**: Download the pre-split data and go directly to training
- **Use your own data**: Follow the full pipeline from step 1
- **Mix and match**: Use pre-generated scenarios but create your own splits

---

## 1. Generate Scenario Definitions

First, generate randomized scenarios (object and robot positions) for simulation.

- **Script:** `scripts/multi_robot_position_scenario_generator.py`
- **How to use:**  
  - Open the script and edit the parameters at the top to set the number of main scenarios and the number of robot positions per scenario (default: 160 scenarios, 90 positions each).
- **Output:**  
  - The generated scenarios will be saved in `output/senarios/` as a JSONL file.  
  - Example: `output/senarios/14400_scenarios_20250606_004922.jsonl`

---

## 2. Run the Simulation to Collect Raw LiDAR Data

Next, use the generated scenarios to run the simulation and collect raw LiDAR scans.

- **Controller:** `bot3/controllers/full_train_multi_scenario_position_controller/full_train_multi_scenario_position_controller.py`
- **How to use:**  
  - Set this controller as the active controller for your TurtleBot3 in the Webots simulation.
  - Start the simulation. The robot will automatically move through all scenarios and positions, logging LiDAR data.
- **Output:**  
  - Raw LiDAR logs will be saved in the `raw_logs/` directory.
  - Example: `raw_logs/yolo1d_scan_2025-06-06_01-11-05.jsonl`

---

## 3. Process Raw Data into Training Images and Labels

Now, convert the raw LiDAR logs into RGB images and YOLO-format labels.  
The process is the same for both "simple fused" and "aligned fused" data; just use the corresponding scripts.

### 3.1. Generate RGB Images (Simple Fused)

- **Script:** `scripts/multi_frame_scenarios_simple_fused_rgb_generator.py`
- **How to use:**  
  - Run the script. It will find the latest log file in `raw_logs/` and generate RGB images.
- **Output:**  
  - Images are saved in:  
    `output/multi_frame/multi_scenarios/simple_fused/X_3_frame_simple_fused_{raw_log_name}`

### 3.2. Generate YOLO Labels (Simple Fused)

- **Script:** `scripts/multi_frame_scenarios_simple_fused_label_generator.py`
- **How to use:**  
  - Run the script. It will process the same log file and generate YOLO-format label files.
- **Output:**  
  - Labels are saved in:  
    `output/multi_frame/multi_scenarios/simple_fused/Y_3_frame_simple_fused_{raw_log_name}`

> **Note:**  
> - Each image/label file is named as `frame_w_x_y_z.png` or `.txt`, where `w, x, y, z` are frame and scenario indices.
> - **Pre-generated data available**: If you want to skip the data generation steps, you can download pre-generated datasets from our [Google Drive](https://drive.google.com/drive/folders/1stHaVEMeqq-YLNqXO39XQT179cmok1Rd?usp=sharing):
>   - **`raw_logs`**: Raw LiDAR data logs from simulation
>   - **`Train_and_Test_Splited_Data_simpled_fused`**: Pre-split training data for simple fused method
>   - **`Train_and_Test_Splited_Data_aligned_fused`**: Pre-split training data for aligned fused method
>   - **`senarios`**: Generated scenario definitions

---

## 4. Split the Dataset for Training/Validation/Test

Split the generated dataset into training, validation, and test sets.

- **Script:** `scripts/split_dataset.py`
- **How to use:**  
  - Run the script. It will prompt you to:
    - Choose the fusion type (simple or aligned fused)
    - Enter the raw log file timestamp (e.g., `2025-06-06_01-11-05`)
    - Choose whether to use cleaned data
    - Set the split percentages for validation and test (e.g., 10% and 5%)
  - The script will group files by main scenario to ensure related frames stay together.
- **Output:**  
  - Split data will be saved in the same folder as the generated data, e.g.:  
    `output/multi_frame/multi_scenarios/simple_fused/splits_simple_fused_2025-06-06_01-11-05`

> **Example split:**  
> - For our training, we used 85% train, 10% validation, 5% test.

---

## 5. (Optional) Compress and Organize Split Data

Due to the large size of the dataset, you may want to compress and organize the split data for easier training and transfer.

- **Example organization:**
  - Training images: `splits_simple_fused_2025-06-06_01-11-05_images_train.tar.gz`
  - Training labels: `splits_simple_fused_2025-06-06_01-11-05_labels_train.tar.gz`
  - Validation and test: `splits_simple_fused_2025-06-06_01-11-05_val_test.tar.gz`

---

## 6. Train the YOLO Model

- **Jupyter Notebooks:**  
  - Training and evaluation notebooks are provided in:  
    `training_outputs/Training_testing_Notebooks/`
  - Open the relevant notebook to see the training scripts and results.

---

## 7. Test the Trained Model

After training, test your model in the Webots simulation environment.

- **Controller:** `bot3/controllers/robot_controller_test_model/robot_controller_test_model.py`
- **How to use:**
  - Set this controller as the active controller for your TurtleBot3 in Webots
  - **Update the model path** in the controller code to point to your trained model:
    ```python
    # Example path for our trained model:
    self.model_path = os.path.join(project_root, "training_outputs/120_DO_simple_Fused/webots_model_assets_simpel_fused_fulltrain_120/yolov8n_lidar.pt")
    ```
  - Start the simulation and observe real-time object detection
- **Features:**
  - **Live YOLO inference** on 3-frame fused LiDAR data
  - **Dual movement modes**: 
    - **Keyboard mode**: Manual control with arrow keys
    - **Waypoint mode**: Automated navigation through predefined waypoints
  - **Real-time visualization**: Robot position, waypoints, and detected objects
  - **Performance metrics**: Inference speed and detection counts
- **Controls:**
  - **M**: Switch between keyboard and waypoint modes
  - **W**: Start waypoint navigation
  - **S**: Stop waypoint navigation
  - **Arrow keys**: Manual movement (in keyboard mode)
  - **Q**: Quit simulation
- **Output**: Real-time visualization showing robot movement, LiDAR scans, and object detections

---

## Summary

**Pipeline overview:**

1. Generate scenarios: `multi_robot_position_scenario_generator.py`
2. Run simulation: `full_train_multi_scenario_position_controller.py` (Webots)
3. Generate RGB images: `multi_frame_scenarios_simple_fused_rgb_generator.py`
4. Generate labels: `multi_frame_scenarios_simple_fused_label_generator.py`
5. Split dataset: `split_dataset.py`
6. (Optional) Compress/organize splits
7. Train using provided notebooks
8. Test model: `robot_controller_test_model.py` (Webots)

---

**Tip:**  
If you want to use the provided split data (e.g., from Google Drive), you can skip steps 1–5 and start from the split folders.

---

## File Structure After Running the Pipeline

```
project_root/
├── output/
│   ├── senarios/
│   │   └── 14400_scenarios_20250606_004922.jsonl
│   └── multi_frame/
│       └── multi_scenarios/
│           └── simple_fused/
│               ├── X_3_frame_simple_fused_yolo1d_scan_2025-06-06_01-11-05/
│               ├── Y_3_frame_simple_fused_yolo1d_scan_2025-06-06_01-11-05/
│               └── splits_simple_fused_2025-06-06_01-11-05/
│                   ├── images/
│                   │   ├── train/
│                   │   ├── val/
│                   │   └── test/
│                   └── labels/
│                       ├── train/
│                       ├── val/
│                       └── test/
├── raw_logs/
│   └── yolo1d_scan_2025-06-06_01-11-05.jsonl
├── training_outputs/
│   ├── Training_testing_Notebooks/
│   │   └── yolov8n_simplefused_120_epochs_DO.ipynb
│   └── 120_DO_simple_Fused/
│       └── webots_model_assets_simpel_fused_fulltrain_120/
│           ├── yolov8n_lidar.pt
│           ├── yolov8n_lidar.onnx
│           └── yolov8n_lidar.torchscript
└── bot3/
    └── controllers/
        ├── full_train_multi_scenario_position_controller/
        │   └── full_train_multi_scenario_position_controller.py
        └── robot_controller_test_model/
            └── robot_controller_test_model.py
```

---

## Requirements

- Python 3.7+
- Webots (for simulation)
- Required Python packages (see `requirements.txt`)
- TurtleBot3 model in Webots

---

## Troubleshooting

- **No scenarios found:** Make sure you've run the scenario generator first
- **No log files:** Check that the controller is properly set in Webots
- **Missing dependencies:** Install required packages from `requirements.txt`
- **Memory issues:** Consider reducing the number of scenarios or robot positions

---

For more detailed information about the project, see the main [README.md](README.md) file. 