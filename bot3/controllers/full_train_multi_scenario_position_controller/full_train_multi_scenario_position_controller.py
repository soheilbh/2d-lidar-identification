import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from controller import Robot, Motor, Lidar, PositionSensor, Supervisor
import math
import json
from typing import List, Dict
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms
import glob

# Import global config
from bot3.controllers.global_config import (
    movement_mode, TIME_STEP, BASE_SPEED, WAYPOINT_DISTANCE_THRESHOLD, TURN_GAIN,
    SAFETY_MARGIN, MIN_WAYPOINT_SPACING, SCAN_DISTANCE,
    ADAPTIVE_SPEED, RETURN_TO_START,
    OBSTACLE_THRESHOLD, AVOIDANCE_WEIGHT, MIN_FRONT_DIST
)

class ScenarioVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_rotated_rect(self, center, width, height, angle_rad, color, label, show_size=True, rotation_angle_deg=None):
        rect = Rectangle((center[0] - width/2, center[1] - height/2), width, height, angle=0, color=color, alpha=0.5, label=label)
        t = mtransforms.Affine2D().rotate_around(center[0], center[1], angle_rad) + self.ax.transData
        rect.set_transform(t)
        self.ax.add_patch(rect)
        self.ax.plot(center[0], center[1], 'ko', markersize=3, zorder=10)
        if show_size:
            self.ax.text(center[0], center[1], f'{width:.2f}×{height:.2f}', color='black', fontsize=7, ha='center', va='center', zorder=11, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
        if rotation_angle_deg is not None:
            self.ax.text(center[0], center[1]-0.13, f'{rotation_angle_deg:.0f}°', color='black', fontsize=7, ha='center', va='center', zorder=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))

    def update(self, scenario_objects, waypoints=None, robot_pose=None, current_waypoint_index=None, title='Scenario', scenario_json=None):
        self.ax.clear()
        self.ax.set_xlim(-2.25, 2.25)
        self.ax.set_ylim(-2.25, 2.25)
        self.ax.grid(True)
        self.ax.set_title(title)
        self.ax.set_aspect('equal')
        # Draw all walls from scenario_json if provided
        wall_drawn = False
        if scenario_json and 'walls' in scenario_json:
            for wall in scenario_json['walls']:
                width = wall.get('width', 0.4)
                height = wall.get('height', 0.4)
                self.add_rotated_rect(wall['center'], width, height, wall['rotation'], 'dimgray', None, show_size=False)
                wall_drawn = True
        else:
            # fallback: draw from state if not provided
            for obj_name, pos_data in scenario_objects.items():
                if obj_name.endswith('wall'):
                    width = pos_data.get('width', 0.4)
                    height = pos_data.get('height', 0.4)
                    self.add_rotated_rect(pos_data['center'], width, height, pos_data['rotation'], 'dimgray', None, show_size=False)
                    wall_drawn = True
        # Draw door frame from scenario_json if provided
        door_drawn = False
        if scenario_json and 'door_frame' in scenario_json:
            df = scenario_json['door_frame']
            width = df.get('width', 0.1)
            height = df.get('height', 0.5)
            rot_deg = np.degrees(df.get('rotation', 0.0))
            self.add_rotated_rect(df['center'], width, height, df['rotation'], 'brown', None, show_size=True, rotation_angle_deg=rot_deg)
            door_drawn = True
        elif 'door_frame' in scenario_objects:
            df = scenario_objects['door_frame']
            width = df.get('width', 0.1)
            height = df.get('height', 0.5)
            rot_deg = np.degrees(df.get('rotation', 0.0))
            self.add_rotated_rect(df['center'], width, height, df['rotation'], 'brown', None, show_size=True, rotation_angle_deg=rot_deg)
            door_drawn = True
        # Draw movable objects
        for obj_name, pos_data in scenario_objects.items():
            if obj_name in ['box', 'chair', 'desk']:
                width = pos_data.get('width', 0.4)
                height = pos_data.get('height', 0.4)
                color = None
                label = None
                rot_deg = np.degrees(pos_data.get('rotation', 0.0))
                if obj_name == 'box':
                    color = 'blue'
                    label = 'Box'
                elif obj_name == 'chair':
                    color = 'green'
                    label = 'Chair'
                elif obj_name == 'desk':
                    color = 'purple'
                    label = 'Desk'
                if color:
                    self.add_rotated_rect(pos_data['center'], width, height, pos_data['rotation'], color, label, show_size=True, rotation_angle_deg=rot_deg)
        # Draw waypoints and add legend handles for them
        waypoint_handle = None
        target_handle = None
        if waypoints:
            xs = [wp[0] for wp in waypoints]
            ys = [wp[1] for wp in waypoints]
            waypoint_handle, = self.ax.plot(xs, ys, 'o-', color='orange', markersize=3, linewidth=1, label='Waypoints', zorder=5)
            if current_waypoint_index is not None and 0 <= current_waypoint_index < len(waypoints):
                tx, ty = waypoints[current_waypoint_index][0], waypoints[current_waypoint_index][1]
                target_handle, = self.ax.plot(tx, ty, 'ro', markersize=7, label='Target Waypoint', zorder=6)
        if robot_pose is not None:
            robot_size = 0.1
            robot_x = robot_pose[0]
            robot_y = robot_pose[1]
            robot_theta = robot_pose[2]
            dx = robot_size * math.cos(robot_theta)
            dy = robot_size * math.sin(robot_theta)
            self.ax.plot([robot_x, robot_x + dx], [robot_y, robot_y + dy], 'k-', linewidth=2)
            self.ax.plot(robot_x, robot_y, 'ko', markersize=5)
        # Custom legend: only one for wall and one for door frame
        handles = []
        labels = []
        if wall_drawn:
            wall_patch = Rectangle((0,0),1,1, color='dimgray', alpha=0.5, label='Wall')
            handles.append(wall_patch)
            labels.append('Wall')
        if door_drawn:
            door_patch = Rectangle((0,0),1,1, color='brown', alpha=0.5, label='Door Frame')
            handles.append(door_patch)
            labels.append('Door Frame')
        # Add other object handles
        handles += [Rectangle((0,0),1,1, color='blue', alpha=0.5, label='Box'),
                    Rectangle((0,0),1,1, color='green', alpha=0.5, label='Chair'),
                    Rectangle((0,0),1,1, color='purple', alpha=0.5, label='Desk')]
        labels += ['Box', 'Chair', 'Desk']
        # Add waypoints and target waypoint to legend if present
        if waypoint_handle:
            handles.append(waypoint_handle)
            labels.append('Waypoints')
        if target_handle:
            handles.append(target_handle)
            labels.append('Target Waypoint')
        self.ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=7)
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

class E2EController:
    def __init__(self):
        self.setup_robot()
        self.visualizer = ScenarioVisualizer()
        self.pose = [0.0, 0.0, 0]
        self.scenario_data = self.load_scenarios()
        self.num_scenarios = len(self.scenario_data)
        self.current_scenario_index = 0
        self.skip_lidar_log = False  # Add flag for skipping LiDAR logging
        self.set_scenario(self.current_scenario_index)
        self.log_file = self._init_logging()
        print(f"[Logger] Logging to: {self.log_file}")
        self.frame_id = 0

    def setup_robot(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.robot_node = self.robot.getSelf()
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_position_sensor = self.robot.getDevice('left wheel sensor')
        self.right_position_sensor = self.robot.getDevice('right wheel sensor')
        self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor.enable(self.timestep)
        self.lidar = self.robot.getDevice('LDS-01')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.lidar_width = self.lidar.getHorizontalResolution()
        self.lidar_fov = self.lidar.getFov()
        self.angle_increment = self.lidar_fov / self.lidar_width
        self.angles = [(i * self.angle_increment+ np.pi) % (2 * np.pi) for i in range(self.lidar_width)]
        # Object and wall nodes
        self.node_names = ['box', 'chair', 'desk', 'right_wall', 'left_wall_up', 'left_wall_bottom', 'up_wall', 'down_wall']
        self.nodes = {name: self.robot.getFromDef(name) for name in self.node_names}
        # Store initial positions
        self.initial_positions = {}
        for name in ['box', 'chair', 'desk']:
            node = self.nodes[name]
            if node:
                pos = node.getPosition()
                rot = node.getOrientation()
                theta = math.atan2(rot[3], rot[0])
                self.initial_positions[name] = {'position': [pos[0], pos[1]], 'rotation': theta}
                print(f"Initial {name} position: ({pos[0]:.2f}, {pos[1]:.2f}), rotation: {theta:.2f}")

    def _init_logging(self):
        """Initialize logging directory and create log file in the project root logs/ folder."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        logs_dir = os.path.join(project_root, 'raw_logs')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f'yolo1d_scan_{timestamp}.jsonl'
        return os.path.join(logs_dir, log_filename)

    def load_scenarios(self):
        """Load all scenarios from the latest JSONL file in output/senarios/."""
        scenarios_dir = os.path.join(os.path.dirname(__file__), '../../../output/senarios')
        jsonl_files = glob.glob(os.path.join(scenarios_dir, '*.jsonl'))
        if not jsonl_files:
            raise FileNotFoundError(f"No scenario JSONL files found in {scenarios_dir}")
        
        # Get the latest file based on modification time
        latest_file = max(jsonl_files, key=os.path.getmtime)
        print(f"[Scenario Loader] Loading scenarios from: {latest_file}")
        scenarios = []
        with open(latest_file, 'r') as f:
            for line in f:
                scenarios.append(json.loads(line))
        return scenarios

    def set_scenario(self, scenario_index):
        """Apply scenario by moving all objects and setting waypoints from loaded data."""
        scenario = self.scenario_data[scenario_index]
        objects = scenario['objects']
        
        # Move all movable objects
        for obj_name in ['box', 'chair', 'desk']:
            obj = objects.get(obj_name)
            if obj:
                width = obj.get('width')
                height = obj.get('height')
                print(f"[DEBUG] Moving {obj_name}: center={obj['center']}, rotation={obj['rotation']}, width={width}, height={height}")
                self.move_object(obj_name, obj['center'][0], obj['center'][1], obj['rotation'], width, height)
        
        # Move walls according to scenario
        walls = scenario.get('walls', [])
        if walls:
            for wall in walls:
                center = wall['center']
                rotation = wall['rotation']
                width = wall.get('width')
                height = wall.get('height')
                wall_name = wall.get('name')
                print(f"[DEBUG] Moving wall {wall_name}: center={center}, rotation={rotation}, width={width}, height={height}")
                if wall_name:
                    self.move_object(wall_name, center[0], center[1], rotation, width, height)
        
        # Move robot to its position from scenario data
        if 'robot_position' in scenario:
            robot_pos = scenario['robot_position']
            print(f"[DEBUG] Moving robot to position: {robot_pos}")
            # Get the robot's translation and rotation fields
            translation_field = self.robot_node.getField('translation')
            rotation_field = self.robot_node.getField('rotation')
            if translation_field and rotation_field:
                # Set position (x, y, z)
                translation_field.setSFVec3f([robot_pos[0], robot_pos[1], 0.0])
                # Set rotation (axis_x, axis_y, axis_z, angle)
                rotation_field.setSFRotation([0, 0, 1, robot_pos[2]])
                # Reset robot physics to ensure proper update
                self.robot_node.resetPhysics()
                # Update the pose to match the new position
                self.pose = robot_pos.copy()
                print(f"[DEBUG] Robot moved to position: {self.pose}")
        
        self.waypoints = scenario['waypoints']
        self.current_waypoint_index = 0
        print(f"[Scenario] Set scenario {scenario_index} with {len(self.waypoints)} waypoints.")

        # Step simulation multiple times to ensure physics and sensors are updated
        for _ in range(3):  # Step 3 times to ensure proper updates
            self.robot.step(self.timestep)

        # Reset LiDAR to prevent old scan data
        self.lidar.disable()
        self.robot.step(self.timestep)
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.skip_lidar_log = True  # Skip logging for this frame

    def move_object(self, object_name, x, y, rotation=0.0, width=None, height=None):
        node = self.nodes.get(object_name)
        if node:
            translation_field = node.getField('translation')
            rotation_field = node.getField('rotation')
            size_field = node.getField('size')
            if translation_field:
                translation_field.setSFVec3f([x, y, 0.0])
            if rotation_field:
                rotation_field.setSFRotation([0, 0, 1, rotation])
            if size_field and width is not None and height is not None:
                old_size = size_field.getSFVec3f()
                size_field.setSFVec3f([width, height, old_size[2]])
                new_size = size_field.getSFVec3f()
                print(f"[DEBUG] Set size for {object_name} from {old_size} to {new_size}")
                if abs(new_size[0] - width) > 1e-6 or abs(new_size[1] - height) > 1e-6:
                    print(f"[WARNING] Tried to set size for {object_name} to {width}x{height}, but actual size is {new_size[0]}x{new_size[1]}")
            # Reset physics to ensure proper update
            node.resetPhysics()
            print(f"Moved {object_name} to ({x:.2f}, {y:.2f}) with rotation {rotation:.2f}" + (f" and size {width}x{height}" if width is not None and height is not None else ""))
            return True
        else:
            print(f"[WARNING] Node for {object_name} not found!")
        return False

    def reset_object(self, object_name):
        """Reset an object to its initial position and rotation."""
        if object_name in self.initial_positions:
            initial = self.initial_positions[object_name]
            return self.move_object(object_name, initial['position'][0], initial['position'][1], initial['rotation'])
        return False

    def update_pose(self):
        """Update robot pose using supervisor position and orientation."""
        position = self.robot_node.getPosition()
        rotation = self.robot_node.getOrientation()
        
        self.pose[0] = position[0]
        self.pose[1] = position[1]
        self.pose[2] = math.atan2(rotation[3], rotation[0])
        
        while self.pose[2] > math.pi:
            self.pose[2] -= 2 * math.pi
        while self.pose[2] < -math.pi:
            self.pose[2] += 2 * math.pi

    def move_to_waypoint(self):
        if hasattr(self, 'mission_complete') and self.mission_complete:
            self.left_motor.setVelocity(0)
            self.right_motor.setVelocity(0)
            return
            
        # Check if there are any waypoints
        if not self.waypoints or len(self.waypoints) == 0:
            print(f"No waypoints in scenario {self.current_scenario_index}, moving to next scenario")
            self.current_scenario_index += 1
            if self.current_scenario_index >= self.num_scenarios:
                print("\n" + "="*50)
                print("MISSION ACCOMPLISHED!")
                print("All scenarios completed successfully.")
                print("="*50 + "\n")
                self.mission_complete = True
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(0)
                return
            self.set_scenario(self.current_scenario_index)
            return
            
        self.update_pose()
        target_x, target_y = self.waypoints[self.current_waypoint_index]
        dx = target_x - self.pose[0]
        dy = target_y - self.pose[1]
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.atan2(dy, dx)
        if distance < WAYPOINT_DISTANCE_THRESHOLD:
            print(f"Reached waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)}")
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                print(f"\nCompleted all waypoints for scenario {self.current_scenario_index}")
                self.mission_complete = True
            return

        # Get LiDAR scan data
        scan = self.lidar.getRangeImage()
        scan = self.replace_inf_with_max(scan, 4.0)
        
        # Define angle ranges to check (in Webots angles)
        # Front sectors: 170° to 190° (centered on 180°)
        # Front-right sectors: 140° to 170° (centered on 155°)
        # Front-left sectors: 190° to 220° (centered on 205°)
        # Right sectors: 110° to 140° (centered on 125°)
        # Left sectors: 220° to 250° (centered on 235°)
        
        # Front sector
        front_start = int(170 * len(scan) / 360)
        front_end = int(190 * len(scan) / 360)
        
        # Front-right sectors
        front_right_start = int(140 * len(scan) / 360)
        front_right_end = int(170 * len(scan) / 360)
        
        # Front-left sectors
        front_left_start = int(190 * len(scan) / 360)
        front_left_end = int(220 * len(scan) / 360)
        
        # Right sectors
        right_start = int(110 * len(scan) / 360)
        right_end = int(140 * len(scan) / 360)
        
        # Left sectors
        left_start = int(220 * len(scan) / 360)
        left_end = int(250 * len(scan) / 360)
        
        # Get minimum distances in each sector
        front_dist = min(scan[front_start:front_end])
        front_right_dist = min(scan[front_right_start:front_right_end])
        front_left_dist = min(scan[front_left_start:front_left_end])
        right_dist = min(scan[right_start:right_end])
        left_dist = min(scan[left_start:left_end])
        
        # # Print debug information
        # print('-------------------------------------------')
        # print(f'Range data at front (170°-190°):      {front_dist:.2f}')
        # print(f'Range data at front-right (140°-170°): {front_right_dist:.2f}')
        # print(f'Range data at front-left (190°-220°):  {front_left_dist:.2f}')
        # print(f'Range data at right (110°-140°):      {right_dist:.2f}')
        # print(f'Range data at left (220°-250°):       {left_dist:.2f}')
        # print('-------------------------------------------')
        
        # Define thresholds
        thr1 = 0.15  # Front threshold
        thr2 = 0.15  # Side thresholds
        
        # Check if path is clear using more detailed sector information
        if (front_dist > thr1 and 
            front_right_dist > thr2 and 
            front_left_dist > thr2 and 
            right_dist > thr2 and 
            left_dist > thr2):
            # Path is clear, move towards waypoint
            angular_error = target_angle - self.pose[2]
            while angular_error > math.pi:
                angular_error -= 2 * math.pi
            while angular_error < -math.pi:
                angular_error += 2 * math.pi
                
            # Calculate base speed
            base_speed = min(BASE_SPEED, 6.0)
            if ADAPTIVE_SPEED:
                base_speed = min(base_speed, distance * 1.5)
            if abs(angular_error) > math.pi/4:
                base_speed *= 0.5
                
            # Set motor speeds for forward movement with turning
            left_speed = min(base_speed - TURN_GAIN * angular_error, 6.0)
            right_speed = min(base_speed + TURN_GAIN * angular_error, 6.0)
        else:
            # Obstacle detected, decide rotation direction based on sector information
            base_speed = min(BASE_SPEED, 6.0)
            if front_right_dist < front_left_dist:
                # More space on the left, rotate counter-clockwise
                left_speed = base_speed
                right_speed = -base_speed * 0.5
            else:
                # More space on the right, rotate clockwise
                left_speed = -base_speed * 0.5
                right_speed = base_speed
            
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def replace_inf_with_max(self, scan, max_distance=4.0):
        scan = np.array(scan)
        scan[~np.isfinite(scan)] = max_distance
        return scan.tolist()

    def log_lidar_data(self, frame_id, raw_scan, angles, pose, filename, object_details=None, scenario_number=1):
        """
        Log LiDAR data in YOLO 1D format, rounding all float values to two decimal places.
        """
        def round2(val):
            if isinstance(val, float):
                return round(val, 2)
            elif isinstance(val, list):
                return [round2(v) for v in val]
            elif isinstance(val, dict):
                return {k: round2(v) for k, v in val.items()}
            else:
                return val

        # Get current scenario data
        scenario = self.scenario_data[scenario_number]
        objects = scenario['objects']
        
        # Create object details dictionary
        object_details = {}
        for obj_name, obj_data in objects.items():
            object_details[obj_name] = {
                "center": round2(obj_data["center"]),
                "rotation": round2(obj_data["rotation"]),
                "width": round2(obj_data["width"]),
                "height": round2(obj_data["height"]),
                "corners": round2(obj_data["corners"])
            }
        
        # Add door frame
        door_frame = scenario['door_frame']
        object_details['door_frame'] = {
            "center": round2(door_frame["center"]),
            "rotation": round2(door_frame["rotation"]),
            "width": round2(door_frame["width"]),
            "height": round2(door_frame["height"]),
            "corners": round2(door_frame["corners"])
        }
        
        log_entry = {
            "frame_id": frame_id,
            "raw_scan": round2(raw_scan),
            "pose": round2(pose),
            "object_details": object_details,
            "scenario_number": scenario_number,
            "main_scenario_id": scenario.get("main_scenario_id", scenario_number)  # Add main scenario ID
        }
        
        with open(filename, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

    def get_current_world_state(self):
        # Returns a dict like the scenario JSONL, but with current values from Webots
        state = {}
        for name, node in self.nodes.items():
            if node:
                pos = node.getField('translation').getSFVec3f()
                rot = node.getField('rotation').getSFRotation()
                size = node.getField('size').getSFVec3f()
                state[name] = {
                    'center': [pos[0], pos[1]],
                    'rotation': rot[3],  # assuming [0,0,1,theta]
                    'width': size[0],
                    'height': size[1]
                }
        return state

    def run(self):
        """Main control loop (refactored for JSONL scenarios)."""
        print("\nStarting scenario-driven run...")
        self.mission_complete = False
        while self.robot.step(self.timestep) != -1:
            current_time = self.robot.getTime()
            
            # Skip frame if LiDAR was just reset
            if self.skip_lidar_log:
                self.skip_lidar_log = False
                continue
                
            # Log data
            object_details = {}  # Optionally fill with get_object_details if needed
            scan = self.lidar.getRangeImage()
            scan = self.replace_inf_with_max(scan, 4.0)
            self.log_lidar_data(self.frame_id, scan, self.angles, self.pose, self.log_file, object_details, self.current_scenario_index)
            self.frame_id += 1
            # Move robot
            if movement_mode == 0:
                pass  # Optionally implement Braitenberg mode
            else:
                self.move_to_waypoint()
                current_state = self.get_current_world_state()
                self.visualizer.update(current_state, self.waypoints, self.pose, self.current_waypoint_index, title=f'Scenario {self.current_scenario_index}', scenario_json=self.scenario_data[self.current_scenario_index])
                if hasattr(self, 'mission_complete') and self.mission_complete:
                    print(f"\nScenario {self.current_scenario_index} completed. Moving to next scenario...")
                    self.current_scenario_index += 1
                    if self.current_scenario_index >= self.num_scenarios:
                        print("\n" + "="*50)
                        print("MISSION ACCOMPLISHED!")
                        print("All scenarios completed successfully.")
                        print("="*50 + "\n")
                        final_log = {
                            "frame_id": self.frame_id,
                            "status": "MISSION_COMPLETE",
                            "total_frames": self.frame_id,
                            "completion_time": current_time,
                            "final_pose": self.pose,
                            "scenarios_completed": self.num_scenarios
                        }
                        with open(self.log_file, 'a') as f:
                            json.dump(final_log, f)
                            f.write('\n')
                        self.left_motor.setVelocity(0)
                        self.right_motor.setVelocity(0)
                        plt.close('all')
                        return
                    self.set_scenario(self.current_scenario_index)
                    self.mission_complete = False

if __name__ == "__main__":
    controller = E2EController()
    controller.run() 