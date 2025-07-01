"""Simple keyboard-controlled robot controller with modular design."""

import sys
import os
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import enum

# Add the controllers directory to the path so we can import global_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from controller import Robot, Motor, Keyboard, Supervisor
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from ultralytics import YOLO

# Import global config with fallback
try:
    from global_config import (
        TIME_STEP, BASE_SPEED, TURN_GAIN
    )
    # Override with maximum speed values for waypoint navigation
    BASE_SPEED = 6.67  # Override to maximum TurtleBot3 speed
    TURN_GAIN = 0.5   # Override for better control
    print("✅ Using global_config with speed overrides for maximum waypoint performance")
except ImportError:
    print("Warning: Could not import global_config, using default values")
    TIME_STEP = 64
    BASE_SPEED = 6.67  # Set to exact TurtleBot3 maximum for absolute fastest speed
    TURN_GAIN = 0.5  # Reduced from 2.5 to 0.5 for slower turning

# Keyboard mode parameters (half speed for better control)
KEYBOARD_SPEED = BASE_SPEED * 0.4  # Half speed for keyboard mode
KEYBOARD_TURN_GAIN = TURN_GAIN * 0.4  # Half turn gain for keyboard mode


class MovementMode(enum.Enum):
    """Enumeration for different movement modes."""
    KEYBOARD = "keyboard"
    WAYPOINT = "waypoint"


@dataclass
class Waypoint:
    """Data class to hold waypoint information."""
    x: float
    y: float
    tolerance: float = 0.1  # Distance tolerance to consider waypoint reached
    reached: bool = False


@dataclass
class RobotState:
    """Data class to hold robot state information."""
    current_speed: float = 0.0
    current_rotation: float = 0.0
    is_running: bool = True
    pose: list = None
    movement_mode: MovementMode = MovementMode.KEYBOARD
    current_waypoint_index: int = 0
    
    def __post_init__(self):
        if self.pose is None:
            self.pose = [0.0, 0.0, 0.0]


class RGBBuffer:
    """RGB buffer for live LiDAR visualization like the RGB generator."""
    
    def __init__(self, buffer_size=3):
        self.buffer_size = buffer_size
        self.binary_arrays = deque(maxlen=buffer_size)
        
        # Initialize YOLO model
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '../../../'))
            self.model_path = os.path.join(project_root, "training_outputs/120_DO_simple_Fused/webots_model_assets_simpel_fused_fulltrain_120/yolov8n_lidar.pt")
            self.model = YOLO(self.model_path, task='detect', verbose=False)
            self.model_loaded = True
            print(f"✅ YOLO model loaded")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            self.model_loaded = False
    
    def add_frame(self, raw_scan, angles):
        """Add a new frame to buffer (FIFO)"""
        binary_array = np.zeros((64, 360), dtype=np.uint8)
        angle_indices = (np.round(np.degrees(angles)) % 360).astype(np.uint16)
        distance_indices = np.clip((np.array(raw_scan) * 15.75), 0, 63).astype(np.uint8)
        binary_array[distance_indices, angle_indices] = 255
        self.binary_arrays.append(binary_array)
    
    def get_rgb_image(self):
        """Get RGB image from current buffer state"""
        if len(self.binary_arrays) < self.buffer_size:
            return None
        
        rgb_image = np.zeros((64, 384, 3), dtype=np.uint8)
        rgb_image[:, :360, 0] = self.binary_arrays[0]  # R channel
        rgb_image[:, :360, 1] = self.binary_arrays[1]  # G channel
        rgb_image[:, :360, 2] = self.binary_arrays[2]  # B channel
        return rgb_image
    
    def run_inference(self, rgb_image):
        """Run YOLO inference on RGB image and return detections"""
        if not self.model_loaded or rgb_image is None:
            return []
        
        try:
            import time
            start_time = time.time()
            
            results = self.model(rgb_image, imgsz=[64, 384], verbose=False)
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            detections = []
            
            for r in results:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                boxes_xywh = r.boxes.xywh.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                names = r.names

                for (x1, y1, x2, y2), (cx, cy, w, h), conf, class_id in zip(boxes_xyxy, boxes_xywh, confs, class_ids):
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (cx, cy),
                        'size': (w, h),
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': names[class_id]
                    })
            
            # Store performance metrics
            self.last_inference_time = inference_time
            self.last_detection_count = len(detections)
            self.last_detection_classes = [d['class_name'] for d in detections]
            
            return detections
        except Exception as e:
            print(f"❌ Inference error: {e}")
            return []
    
    def convert_detection_to_world_coords(self, detection, robot_pose):
        """Convert YOLO detection box to world coordinates relative to robot"""
        robot_x, robot_y, robot_theta = robot_pose
        
        # Get detection center in image coordinates
        cx_img, cy_img = detection['center']
        
        # Convert image coordinates to normalized coordinates (0-1)
        # Image is 64x384, but LiDAR data is only in first 360 columns
        cx_norm = cx_img / 360.0  # Normalize by 360 (LiDAR data width)
        cy_norm = cy_img / 64.0   # Normalize by 64 (LiDAR data height)
        
        # Convert normalized coordinates to LiDAR coordinates
        # cx_norm: 0-1 represents 0-359 degrees (clockwise from robot's right)
        # cy_norm: 0-1 represents 0-63 distance bins (0-4m)
        
        # Convert angle (normalized to degrees) - LiDAR is CW
        angle_deg = cx_norm * 359.0  # 0-359 degrees (CW)
        
        # Convert distance (normalized to meters)
        distance = cy_norm * 4.0  # 0-4 meters
        
        # Convert to radians
        angle_rad = np.radians(angle_deg)
        
        # Reverse the process from post_process.py
        # In post_process.py: angle = (360 - angle) % 360 for CW conversion
        # So we need to reverse this: original_angle = (360 - lidar_angle) % 360
        original_angle_deg = (360 - angle_deg) % 360
        original_angle_rad = np.radians(original_angle_deg)
        
        # Calculate robot-relative coordinates from distance and angle
        # This is the reverse of: angle = np.degrees(np.arctan2(rel_y, rel_x))
        rel_x = distance * np.cos(original_angle_rad)
        rel_y = distance * np.sin(original_angle_rad)
        
        # Convert to world coordinates using the inverse of the rotation matrix
        # In post_process.py: rel_x = cos(theta)*global_rel_x + sin(theta)*global_rel_y
        #                    rel_y = -sin(theta)*global_rel_x + cos(theta)*global_rel_y
        # So to get global coordinates: global_rel_x = cos(theta)*rel_x - sin(theta)*rel_y
        #                                global_rel_y = sin(theta)*rel_x + cos(theta)*rel_y
        global_rel_x = np.cos(robot_theta) * rel_x - np.sin(robot_theta) * rel_y
        global_rel_y = np.sin(robot_theta) * rel_x + np.cos(robot_theta) * rel_y
        
        # Add robot position to get world coordinates
        world_x = robot_x + global_rel_x
        world_y = robot_y + global_rel_y
        
        return {
            'world_center': (world_x, world_y),
            'angle_deg': angle_deg,
            'distance': distance,
            'confidence': detection['confidence'],
            'class_name': detection['class_name']
        }


class SimpleVisualizer:
    """Simple visualization showing only robot and its heading."""
    
    def __init__(self):
        plt.ion()
        # Create figure with gridspec for better control of subplot sizes
        self.fig = plt.figure(figsize=(6, 6))  # Reduced from (8, 10) to (6, 8)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 0.5])  # Robot plot gets 2/3, RGB gets 1/3
        self.ax1 = self.fig.add_subplot(gs[0])  # Robot position (top, larger)
        self.ax2 = self.fig.add_subplot(gs[1])  # RGB LiDAR (bottom, smaller)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Initialize RGB buffer
        self.rgb_buffer = RGBBuffer(buffer_size=3)
    
    def update(self, robot_pose, lidar_scan, angles, waypoint_navigator=None, movement_mode=None, title='Robot Position'):
        """Update visualization with robot position and heading."""
        # Clear all subplots
        self.ax1.clear()
        self.ax2.clear()
        
        # Robot position subplot
        self.ax1.set_xlim(-3, 3)
        self.ax1.set_ylim(-3, 3)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title(title)
        self.ax1.set_aspect('equal')
        
        # Draw very light boundary walls
        wall_alpha = 0.2  # More transparent (reduced from 0.1 to 0.05)
        wall_color = 'purple'  # Changed from gray to purple
        
        # Right wall (x = 2.0) - vertical line from y=-2 to y=2
        self.ax1.plot([2.0, 2.0], [-2.0, 2.0], color=wall_color, alpha=wall_alpha, linewidth=2, label='Boundary')
        
        # Left wall (x = -2.0) - vertical line from y=-2 to y=2
        self.ax1.plot([-2.0, -2.0], [-2.0, 2.0], color=wall_color, alpha=wall_alpha, linewidth=2)
        
        # Top wall (y = 2.0) - horizontal line from x=-2 to x=2
        self.ax1.plot([-2.0, 2.0], [2.0, 2.0], color=wall_color, alpha=wall_alpha, linewidth=2)
        
        # Bottom wall (y = -2.0) - horizontal line from x=-2 to x=2
        self.ax1.plot([-2.0, 2.0], [-2.0, -2.0], color=wall_color, alpha=wall_alpha, linewidth=2)
        
        # Draw waypoints if available
        if waypoint_navigator and waypoint_navigator.waypoints:
            waypoints = waypoint_navigator.waypoints
            current_index = waypoint_navigator.current_waypoint_index
            
            # Draw all waypoints
            for i, wp in enumerate(waypoints):
                if i < current_index:
                    # Reached waypoints - green
                    color = 'green'
                    marker = 'o'
                    alpha = 0.6
                elif i == current_index and waypoint_navigator.navigation_active:
                    # Current waypoint - red
                    color = 'red'
                    marker = 's'
                    alpha = 0.8
                else:
                    # Future waypoints - blue
                    color = 'blue'
                    marker = 'o'
                    alpha = 0.4
                
                self.ax1.scatter(wp.x, wp.y, c=color, marker=marker, s=100, alpha=alpha, zorder=5)
                
                # Add waypoint number
                self.ax1.text(wp.x, wp.y + 0.1, f'WP{i+1}', 
                             ha='center', va='bottom', fontsize=8, weight='bold',
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            # Draw path between waypoints
            if len(waypoints) > 1:
                wp_x = [wp.x for wp in waypoints]
                wp_y = [wp.y for wp in waypoints]
                self.ax1.plot(wp_x, wp_y, 'k--', alpha=0.3, linewidth=1, zorder=1)
        
        # Draw robot position
        robot_x, robot_y, robot_theta = robot_pose
        
        # Robot body (circle)
        robot_size = 0.1
        circle = plt.Circle((robot_x, robot_y), robot_size, color='blue', alpha=0.7)
        self.ax1.add_patch(circle)
        
        # Robot heading (arrow)
        heading_length = 0.2
        dx = heading_length * math.cos(robot_theta)
        dy = heading_length * math.sin(robot_theta)
        self.ax1.arrow(robot_x, robot_y, dx, dy, head_width=0.05, head_length=0.05, 
                     fc='red', ec='red', alpha=0.8)
        
        # Add text with position and angle
        angle_deg = math.degrees(robot_theta)
        self.ax1.text(robot_x, robot_y + 0.3, f'Pos: ({robot_x:.2f}, {robot_y:.2f})\nAngle: {angle_deg:.1f}°', 
                    ha='center', va='center', fontsize=6, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add navigation status if in waypoint mode
        if waypoint_navigator and movement_mode == MovementMode.WAYPOINT:
            nav_status = waypoint_navigator.get_navigation_status()
            mode_text = f"Mode: {movement_mode.value.upper()}"
            status_text = f"{mode_text}\n{nav_status}"
            self.ax1.text(0.02, 0.98, status_text, 
                         transform=self.ax1.transAxes, fontsize=8, 
                         verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        elif movement_mode:
            mode_text = f"Mode: {movement_mode.value.upper()}"
            self.ax1.text(0.02, 0.98, mode_text, 
                         transform=self.ax1.transAxes, fontsize=8, 
                         verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # RGB LiDAR visualization subplot
        self.ax2.set_title('Live RGB LiDAR (3-Frame Buffer) + YOLO Detections')
        
        # Add current frame to RGB buffer
        if lidar_scan and angles:
            self.rgb_buffer.add_frame(lidar_scan, angles)
        
        # Get RGB image from buffer
        rgb_image = self.rgb_buffer.get_rgb_image()
        if rgb_image is not None:
            # Run YOLO inference
            detections = self.rgb_buffer.run_inference(rgb_image)
            
            # Display RGB image
            self.ax2.imshow(rgb_image, aspect='auto', origin='lower')
            self.ax2.set_xlabel('Angle (degrees)')
            self.ax2.set_ylabel('Distance (scaled)')
            
            # Draw detection boxes in RGB plot
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                conf = detection['confidence']
                class_name = detection['class_name']
                
                # Draw rectangle
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='lime', facecolor='none', alpha=0.8)
                self.ax2.add_patch(rect)
                
                # Add label
                label = f"{class_name} {conf:.2f}"
                self.ax2.text(x1, y1-2, label, fontsize=6, color='lime', weight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
            
            # Convert detections to world coordinates and draw in robot plot
            detection_handles = []
            for detection in detections:
                world_detection = self.rgb_buffer.convert_detection_to_world_coords(detection, robot_pose)
                world_x, world_y = world_detection['world_center']
                conf = world_detection['confidence']
                class_name = world_detection['class_name']
                
                # Draw detection point in world coordinates
                point = self.ax1.scatter(world_x, world_y, c='lime', marker='o', s=50, alpha=0.8)
                detection_handles.append(point)
                
                # Add label
                label = f"{class_name}\n{conf:.2f}"
                self.ax1.text(world_x, world_y + 0.05, label, 
                             fontsize=6, color='lime', weight='bold', ha='center',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
            
            # Add performance metrics and useful information
            perf_text = f'Live YOLO Detection'
            
            # Add performance metrics if available
            if hasattr(self.rgb_buffer, 'last_inference_time'):
                perf_text += f'\nSpeed: {self.rgb_buffer.last_inference_time:.1f}ms'
                if hasattr(self.rgb_buffer, 'last_detection_classes') and self.rgb_buffer.last_detection_classes:
                    class_counts = {}
                    for cls in self.rgb_buffer.last_detection_classes:
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                    class_str = ', '.join([f"{count} {cls}" for cls, count in class_counts.items()])
                    perf_text += f'\nObjects: {class_str}'
                else:
                    perf_text += f'\nObjects: None detected'
            
            self.ax2.text(0.02, 0.98, perf_text, 
                         transform=self.ax2.transAxes, fontsize=6, 
                         verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            # Show message when buffer is not full
            self.ax2.text(0.5, 0.5, 'Buffer filling...\n(Need 3 frames)', 
                         transform=self.ax2.transAxes, ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            self.ax2.set_xlim(0, 1)
            self.ax2.set_ylim(0, 1)
        
        # Update display
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


class LidarProcessor:
    """Handles LiDAR data processing exactly like full_train_multi_scenario_position_controller."""
    
    def __init__(self, lidar_device, robot_node):
        self.lidar = lidar_device
        self.robot_node = robot_node
        self.lidar_width = self.lidar.getHorizontalResolution()
        self.lidar_fov = self.lidar.getFov()
        self.angle_increment = self.lidar_fov / self.lidar_width
        self.angles = [(i * self.angle_increment + np.pi) % (2 * np.pi) for i in range(self.lidar_width)]
    
    def get_scan_data(self) -> list:
        """Get LiDAR scan data with infinite values replaced."""
        scan = self.lidar.getRangeImage()
        return self.replace_inf_with_max(scan, 4.0)
    
    def replace_inf_with_max(self, scan, max_distance=4.0):
        """Replace infinite values in scan with maximum distance."""
        scan = np.array(scan)
        scan[~np.isfinite(scan)] = max_distance
        return scan.tolist()
    
    def update_pose(self, pose):
        """Update robot pose using supervisor position and orientation."""
        position = self.robot_node.getPosition()
        rotation = self.robot_node.getOrientation()
        
        pose[0] = position[0]
        pose[1] = position[1]
        pose[2] = math.atan2(rotation[3], rotation[0])
        
        while pose[2] > math.pi:
            pose[2] -= 2 * math.pi
        while pose[2] < -math.pi:
            pose[2] += 2 * math.pi


class MotorController:
    """Handles motor control and speed calculations."""
    
    def __init__(self, left_motor: Motor, right_motor: Motor):
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.max_speed = 6.67  # Set to absolute maximum TurtleBot3 speed
    
    def set_speeds(self, speed: float, rotation: float) -> None:
        """Set motor speeds with safety limits."""
        left_speed = speed - rotation
        right_speed = speed + rotation
        
        # Apply speed limits
        left_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        right_speed = max(-self.max_speed, min(self.max_speed, right_speed))
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
    
    def stop(self) -> None:
        """Stop both motors."""
        self.set_speeds(0.0, 0.0)


class KeyboardHandler:
    """Handles keyboard input and command mapping."""
    
    def __init__(self, keyboard_device):
        self.keyboard = keyboard_device
        self.command_map = self._create_command_map()
    
    def _create_command_map(self) -> Dict[int, str]:
        """Create mapping of key codes to commands."""
        return {
            315: 'forward',      # Up arrow
            317: 'backward',     # Down arrow
            316: 'left',         # Left arrow
            314: 'right',        # Right arrow
            ord(' '): 'stop',    # Space
            ord('Q'): 'quit',    # Q
            ord('q'): 'quit',
            ord('M'): 'switch_mode',  # M - Switch movement mode
            ord('m'): 'switch_mode',
            ord('W'): 'start_waypoints',  # W - Start waypoint navigation
            ord('w'): 'start_waypoints',
            ord('S'): 'stop_waypoints',   # S - Stop waypoint navigation
            ord('s'): 'stop_waypoints',
        }
    
    def get_command(self) -> Optional[str]:
        """Get command from keyboard input."""
        key = self.keyboard.getKey()
        return self.command_map.get(key) if key != -1 else None


class WaypointNavigator:
    """Handles waypoint navigation logic."""
    
    def __init__(self, waypoints: List[Waypoint] = None):
        self.waypoints = waypoints or self._get_default_waypoints()
        self.current_waypoint_index = 0
        self.navigation_active = False
        self.position_tolerance = 0.1
        self.angle_tolerance = 0.1  # radians (~5.7 degrees)
        self.max_speed = BASE_SPEED * 1.0  # Use full BASE_SPEED for maximum speed
        self.turn_gain = 3.0  # High turn gain for fast direction changes
    
    def _get_default_waypoints(self) -> List[Waypoint]:
        """Get default waypoints for demonstration."""
        return [
            Waypoint(1, 0, 0.1),
            Waypoint(0, 1, 0.1),
            Waypoint(-1, 0, 0.1),
            Waypoint(-2.5, 0, 0.1),
            Waypoint(-1, 0, 0.1),
            Waypoint(-1.0, -1.0, 0.1),
            Waypoint(0.0, 0.0, 0.1),  # Return to start
        ]
    
    def set_waypoints(self, waypoints: List[Waypoint]) -> None:
        """Set new waypoints and reset navigation."""
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.navigation_active = True
        for wp in self.waypoints:
            wp.reached = False
    
    def start_navigation(self) -> None:
        """Start waypoint navigation."""
        self.navigation_active = True
        self.current_waypoint_index = 0
        for wp in self.waypoints:
            wp.reached = False
        print(f"🚀 Starting waypoint navigation with {len(self.waypoints)} waypoints")
    
    def stop_navigation(self) -> None:
        """Stop waypoint navigation."""
        self.navigation_active = False
        print("⏹️ Stopping waypoint navigation")
    
    def get_current_waypoint(self) -> Optional[Waypoint]:
        """Get current waypoint."""
        if not self.navigation_active or self.current_waypoint_index >= len(self.waypoints):
            return None
        return self.waypoints[self.current_waypoint_index]
    
    def is_navigation_complete(self) -> bool:
        """Check if all waypoints have been reached."""
        return self.current_waypoint_index >= len(self.waypoints)
    
    def calculate_movement(self, robot_pose: List[float]) -> Tuple[float, float]:
        """Calculate speed and rotation to reach current waypoint."""
        if not self.navigation_active:
            return 0.0, 0.0
        
        current_waypoint = self.get_current_waypoint()
        if not current_waypoint:
            return 0.0, 0.0
        
        robot_x, robot_y, robot_theta = robot_pose
        wp_x, wp_y = current_waypoint.x, current_waypoint.y
        
        # Calculate distance and angle to waypoint
        dx = wp_x - robot_x
        dy = wp_y - robot_y
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.atan2(dy, dx)
        
        # Check if waypoint reached
        if distance <= current_waypoint.tolerance:
            current_waypoint.reached = True
            self.current_waypoint_index += 1
            if self.is_navigation_complete():
                print("✅ All waypoints reached!")
                self.navigation_active = False
                return 0.0, 0.0
            else:
                print(f"🎯 Reached waypoint {self.current_waypoint_index-1}, moving to next...")
                return self.calculate_movement(robot_pose)  # Recursive call for next waypoint
        
        # Calculate angle difference
        angle_diff = target_angle - robot_theta
        
        # Normalize angle difference to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Calculate speed and rotation
        speed = min(self.max_speed, distance * 4.0)  # High multiplier for fast approach
        speed = max(speed, self.max_speed * 0.6)  # Minimum 60% of max speed to avoid slowing down near waypoints
        
        # If angle difference is large, reduce speed and focus on turning
        if abs(angle_diff) > self.angle_tolerance:
            speed *= 0.8  # Minimal speed reduction for maximum speed turning
            rotation = self.turn_gain * angle_diff
        else:
            rotation = self.turn_gain * angle_diff * 1.0  # Full responsiveness
        
        return speed, rotation
    
    def get_navigation_status(self) -> str:
        """Get current navigation status string."""
        if not self.navigation_active:
            return "Navigation: Inactive"
        
        if self.is_navigation_complete():
            return "Navigation: Complete"
        
        current_wp = self.get_current_waypoint()
        if current_wp:
            return f"Navigation: {self.current_waypoint_index + 1}/{len(self.waypoints)} - ({current_wp.x:.1f}, {current_wp.y:.1f})"
        
        return "Navigation: Unknown"


class RobotController:
    """Main robot controller class."""
    
    def __init__(self):
        self.robot = None
        self.timestep = TIME_STEP
        self.state = RobotState()
        
        # Initialize components
        self._setup_robot()
        self._setup_components()
        self._print_instructions()
    
    def _setup_robot(self) -> None:
        """Initialize robot and basic devices."""
        self.robot = Supervisor()  # Use Supervisor like full_train controller
        self.timestep = int(self.robot.getBasicTimeStep())
        self.robot_node = self.robot.getSelf()
        
        # Setup motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Setup LiDAR exactly like full_train_multi_scenario_position_controller
        self.lidar = self.robot.getDevice('LDS-01')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.lidar_width = self.lidar.getHorizontalResolution()
        self.lidar_fov = self.lidar.getFov()
        self.angle_increment = self.lidar_fov / self.lidar_width
        self.angles = [(i * self.angle_increment + np.pi) % (2 * np.pi) for i in range(self.lidar_width)]
        
        # Setup keyboard - use Webots Keyboard class directly
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)
    
    def _setup_components(self) -> None:
        """Initialize modular components."""
        self.lidar_processor = LidarProcessor(self.lidar, self.robot_node)
        self.motor_controller = MotorController(self.left_motor, self.right_motor)
        self.keyboard_handler = KeyboardHandler(self.keyboard)
        self.visualizer = SimpleVisualizer()
        self.waypoint_navigator = WaypointNavigator()
    
    def _print_instructions(self) -> None:
        """Print control instructions."""
        print("\n" + "="*50)
        print("KEYBOARD CONTROLS:")
        print("="*50)
        print("MOVEMENT CONTROLS:")
        print("Up Arrow    - Move Forward")
        print("Down Arrow  - Move Backward")
        print("Left Arrow  - Turn Left")
        print("Right Arrow - Turn Right")
        print("Space       - Stop")
        print("")
        print("MODE CONTROLS:")
        print("M           - Switch between Keyboard/Waypoint modes")
        print("W           - Start waypoint navigation")
        print("S           - Stop waypoint navigation")
        print("Q           - Quit")
        print("="*50)
        print(f"Waypoint Mode - Base Speed: {BASE_SPEED}, Turn Gain: {TURN_GAIN}")
        print(f"Keyboard Mode - Base Speed: {KEYBOARD_SPEED}, Turn Gain: {KEYBOARD_TURN_GAIN}")
        print("LiDAR: ENABLED (same as full_train controller)")
        print("Visualization: Robot position, waypoints, and YOLO detections")
        print("IMPORTANT: Click on Webots 3D window for focus!")
        print("="*50 + "\n")
    
    def _execute_command(self, command: str) -> None:
        """Execute a movement command."""
        if command == 'quit':
            self.state.is_running = False
            print("Quitting...")
            return
        
        if command == 'switch_mode':
            if self.state.movement_mode == MovementMode.KEYBOARD:
                self.state.movement_mode = MovementMode.WAYPOINT
                print("🔄 Switched to WAYPOINT mode")
            else:
                self.state.movement_mode = MovementMode.KEYBOARD
                self.waypoint_navigator.stop_navigation()
                print("🔄 Switched to KEYBOARD mode")
            return
        
        if command == 'start_waypoints':
            if self.state.movement_mode == MovementMode.WAYPOINT:
                self.waypoint_navigator.start_navigation()
            else:
                print("⚠️ Switch to waypoint mode first (press M)")
            return
        
        if command == 'stop_waypoints':
            self.waypoint_navigator.stop_navigation()
            return
        
        # Only handle movement commands in keyboard mode
        if self.state.movement_mode != MovementMode.KEYBOARD:
            print(f"⚠️ Movement commands only work in KEYBOARD mode (current: {self.state.movement_mode.value})")
            return
        
        if command == 'stop':
            self.state.current_speed = 0.0
            self.state.current_rotation = 0.0
            self.motor_controller.stop()
            print("Stopped")
            return
        
        # Set movement parameters
        if command == 'forward':
            self.state.current_speed = KEYBOARD_SPEED
            self.state.current_rotation = 0.0
            print("Moving forward")
        elif command == 'backward':
            self.state.current_speed = -KEYBOARD_SPEED
            self.state.current_rotation = 0.0
            print("Moving backward")
        elif command == 'left':
            self.state.current_speed = 0.0
            self.state.current_rotation = -KEYBOARD_SPEED * KEYBOARD_TURN_GAIN
            print("Turning left")
        elif command == 'right':
            self.state.current_speed = 0.0
            self.state.current_rotation = KEYBOARD_SPEED * KEYBOARD_TURN_GAIN
            print("Turning right")
        
        # Apply movement
        self.motor_controller.set_speeds(self.state.current_speed, self.state.current_rotation)
    
    def _handle_waypoint_movement(self) -> None:
        """Handle waypoint navigation movement."""
        if self.state.movement_mode != MovementMode.WAYPOINT:
            return
        
        if not self.waypoint_navigator.navigation_active:
            self.motor_controller.stop()
            return
        
        # Calculate movement from waypoint navigator
        speed, rotation = self.waypoint_navigator.calculate_movement(self.state.pose)
        
        # Apply movement
        self.state.current_speed = speed
        self.state.current_rotation = rotation
        self.motor_controller.set_speeds(speed, rotation)
    
    def _handle_movement_cycle(self) -> None:
        """Handle one movement cycle with brief movement and stop."""
        # Move for a few steps
        for _ in range(5):
            self.robot.step(self.timestep)
        
        # Stop after brief movement (only in keyboard mode)
        if self.state.movement_mode == MovementMode.KEYBOARD:
            self.state.current_speed = 0.0
            self.state.current_rotation = 0.0
            self.motor_controller.stop()
    
    def run(self) -> None:
        """Main control loop."""
        print("Starting robot controller with dual movement modes...")
        
        while self.robot.step(self.timestep) != -1 and self.state.is_running:
            # Update pose like full_train controller
            self.lidar_processor.update_pose(self.state.pose)
            
            # Get LiDAR scan data (same as full_train controller)
            scan = self.lidar_processor.get_scan_data()
            
            # Update visualization
            self.visualizer.update(self.state.pose, scan, self.angles, self.waypoint_navigator, self.state.movement_mode, title='Robot Position and LiDAR Scan')
            
            # Get keyboard command
            command = self.keyboard_handler.get_command()
            
            if command:
                self._execute_command(command)
                if self.state.is_running and self.state.movement_mode == MovementMode.KEYBOARD:
                    self._handle_movement_cycle()
            
            # Handle waypoint movement
            self._handle_waypoint_movement()
            
            # Ensure motors are stopped when no command is active (keyboard mode only)
            if self.state.movement_mode == MovementMode.KEYBOARD and not command:
                self.motor_controller.set_speeds(0.0, 0.0)


def main():
    """Main entry point."""
    try:
        controller = RobotController()
        controller.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
