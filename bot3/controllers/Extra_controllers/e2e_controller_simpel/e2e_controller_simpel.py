import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from controller import Robot, Motor, Lidar, PositionSensor, Supervisor
import math
import json
from typing import List, Dict, NamedTuple, Tuple
import numpy as np
from datetime import datetime

# Import global config
from bot3.controllers.global_config import (
    movement_mode, TIME_STEP, BASE_SPEED, WAYPOINT_DISTANCE_THRESHOLD, TURN_GAIN,
    SAFETY_MARGIN, MIN_WAYPOINT_SPACING, SCAN_DISTANCE,
    ADAPTIVE_SPEED, RETURN_TO_START,
    OBSTACLE_THRESHOLD, AVOIDANCE_WEIGHT, MIN_FRONT_DIST
)

class Object2D:
    """
    Represents a 2D object in the environment.
    """
    def __init__(self, x, y, width, depth, name, rotation=0.0, should_scan=True):
        self.x = x
        self.y = y
        self.width = width
        self.depth = depth
        self.name = name
        self.rotation = rotation
        self.should_scan = should_scan

    def get_dimensions(self):
        """Get the correct width and depth based on rotation."""
        if abs(self.rotation % (2 * math.pi) - math.pi/2) < 0.1 or \
           abs(self.rotation % (2 * math.pi) - 3 * math.pi/2) < 0.1:
            return self.depth, self.width
        return self.width, self.depth

# Define known objects in the environment
KNOWN_OBJECTS = [
    Object2D(x=0, y=1, width=0.4, depth=0.4, name="v_box1"),
    Object2D(x=-1, y=-0.7, width=0.4, depth=0.4, name="v_box2"),
    # Object2D(x=0.7, y=-0.9, width=0.8, depth=1.6, name="desk", rotation=math.pi/2)  # 90 degrees rotation
]

def log_lidar_data(frame_id: int, raw_scan: List[float], angles: List[float], pose: List[float], filename: str):
    """
    Log LiDAR data in YOLO 1D format.
    
    Args:
        frame_id: Frame counter (integer)
        raw_scan: Raw LiDAR scan data (360 points)
        angles: Corresponding angles (not logged since they are constant)
        pose: Robot pose [x, y, theta]
        filename: Output file name
    """
    log_entry = {
        "frame_id": frame_id,
        "raw_scan": raw_scan,
        "pose": pose
    }
    
    with open(filename, 'a') as f:
        json.dump(log_entry, f)
        f.write('\n')

def optimize_waypoints(waypoints: List[List[float]]) -> List[List[float]]:
    """Optimize waypoint sequence and ensure minimum spacing."""
    if not waypoints:
        return []
    
    optimized = [waypoints[0]]
    
    for point in waypoints[1:]:
        last_point = optimized[-1]
        dist = math.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
        
        if dist >= MIN_WAYPOINT_SPACING:
            optimized.append(point)
    
    return optimized

class ObjectState:
    def __init__(self, obj: Object2D):
        self.obj = obj
        self.scanned = False
        self.last_scan_time = 0
        self.scan_quality = 0
        self.scan_points = []

class E2EController:
    def __init__(self):
        # Initialize as Supervisor instead of Robot
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Get robot node for position tracking
        self.robot_node = self.robot.getSelf()
        
        # Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Position sensors
        self.left_position_sensor = self.robot.getDevice('left wheel sensor')
        self.right_position_sensor = self.robot.getDevice('right wheel sensor')
        self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor.enable(self.timestep)
        
        # LiDAR
        self.lidar = self.robot.getDevice('LDS-01')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        
        # Get LiDAR properties
        self.lidar_width = self.lidar.getHorizontalResolution()
        self.lidar_fov = self.lidar.getFov()
        self.angle_increment = self.lidar_fov / self.lidar_width
        self.angles = [(i * self.angle_increment+ np.pi) % (2 * np.pi) for i in range(self.lidar_width)]
        
        # Initialize flags
        self.completion_message_shown = False
        self.mission_complete = False
        
        # Initialize object states
        self.object_states = [ObjectState(obj) for obj in KNOWN_OBJECTS]
        self.start_position = [0.0, 0.0]
        
        # Initialize pose tracking
        self.pose = [0.0, 0.0, 0]  # x, y, theta
        
        # Generate initial waypoints
        self.waypoints = self.generate_exploration_waypoints()
        self.current_waypoint_index = 0
        
        # Initialize logging
        self.log_file = self._init_logging()
        print(f"[Logger] Logging to: {self.log_file}")
        
        self.frame_id = 0  # Add frame counter

    def _init_logging(self):
        """Initialize logging directory and create log file in the project root logs/ folder."""
        # Get the project root (two levels up from this file)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        logs_dir = os.path.join(project_root, 'raw_logs')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f'yolo1d_scan_{timestamp}.jsonl'
        return os.path.join(logs_dir, log_filename)

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
        """Enhanced waypoint navigation with object state tracking."""
        if self.mission_complete:
            self.left_motor.setVelocity(0)
            self.right_motor.setVelocity(0)
            return
            
        self.update_pose()
        
        target_x, target_y, waypoint_owner = self.waypoints[self.current_waypoint_index]
        
        dx = target_x - self.pose[0]
        dy = target_y - self.pose[1]
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.atan2(dy, dx)
        
        # Get raw LiDAR scan
        scan = self.lidar.getRangeImage()
        
        # Check if we've reached the waypoint
        if distance < WAYPOINT_DISTANCE_THRESHOLD:
            total_waypoints_per_object = 13
            total_objects = len(self.object_states)
            total_object_waypoints = total_objects * total_waypoints_per_object
            
            if waypoint_owner == "return" or waypoint_owner == "start":
                if waypoint_owner == "start":
                    dx_to_start = self.pose[0] - self.start_position[0]
                    dy_to_start = self.pose[1] - self.start_position[1]
                    distance_to_start = math.sqrt(dx_to_start*dx_to_start + dy_to_start*dy_to_start)
                    
                    if distance_to_start < WAYPOINT_DISTANCE_THRESHOLD:
                        print("Mission complete! All objects scanned and returned to start position.")
                        print(f"Final position: ({self.pose[0]:.3f}, {self.pose[1]:.3f})")
                        self.left_motor.setVelocity(0)
                        self.right_motor.setVelocity(0)
                        self.mission_complete = True
                        return
                    else:
                        print(f"Approaching start position... Current distance: {distance_to_start:.3f}m")
                else:
                    print("Returning to start position...")
            else:
                current_obj = next((obj for obj in self.object_states if obj.obj.name == waypoint_owner), None)
                if current_obj is not None:
                    obj_waypoints = [w for w in self.waypoints if w[2] == waypoint_owner]
                    waypoint_in_circle = obj_waypoints.index(self.waypoints[self.current_waypoint_index]) + 1
                    
                    if waypoint_in_circle == 1:
                        print(f"\nStarting exploration around {current_obj.obj.name} at ({current_obj.obj.x:.2f}, {current_obj.obj.y:.2f})")
                        print(f"Total waypoints for this object: {total_waypoints_per_object}")
                    
                    print(f"Reached waypoint {waypoint_in_circle}/{total_waypoints_per_object} for object {current_obj.obj.name}")
            
            if not self.mission_complete:
                self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
                return
        
        # Calculate angular error
        angular_error = target_angle - self.pose[2]
        while angular_error > math.pi:
            angular_error -= 2 * math.pi
        while angular_error < -math.pi:
            angular_error += 2 * math.pi
        
        # Calculate base speed
        base_speed = BASE_SPEED if not ADAPTIVE_SPEED else min(BASE_SPEED, distance * 2.0)
        
        # Adjust speed based on angle error
        if abs(angular_error) > math.pi/4:
            base_speed *= 0.5
        
        left_speed = base_speed - TURN_GAIN * angular_error
        right_speed = base_speed + TURN_GAIN * angular_error
        
        # Simple obstacle avoidance using raw LiDAR data
        min_dist = min(scan[int(0.25 * self.lidar_width):int(0.75 * self.lidar_width)])
        if min_dist < MIN_FRONT_DIST:
            left_speed = -BASE_SPEED * 0.5
            right_speed = -BASE_SPEED * 0.5
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def generate_object_exploration_waypoints(self, obj: Object2D) -> List[List[float]]:
        """Generate waypoints in a rectangular path around the object."""
        width, depth = obj.get_dimensions()
        print(f"[DEBUG] {obj.name}: rotation={obj.rotation:.2f} width={obj.width} depth={obj.depth} -> used width={width} depth={depth}")
        half_width = (width / 2) + SAFETY_MARGIN
        half_depth = (depth / 2) + SAFETY_MARGIN
        
        corners = [
            [obj.x + half_width, obj.y + half_depth],
            [obj.x - half_width, obj.y + half_depth],
            [obj.x - half_width, obj.y - half_depth],
            [obj.x + half_width, obj.y - half_depth]
        ]
        
        waypoints = []
        points_per_side = 2
        
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            
            for t in range(points_per_side + 1):
                alpha = t / (points_per_side + 1)
                point = [
                    start[0] + (end[0] - start[0]) * alpha,
                    start[1] + (end[1] - start[1]) * alpha
                ]
                waypoints.append([point[0], point[1], obj.name])
        
        # Find closest waypoint
        min_dist = float('inf')
        closest_index = 0
        for i, point in enumerate(waypoints):
            dx = point[0] - self.pose[0]
            dy = point[1] - self.pose[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        
        waypoints = waypoints[closest_index:] + waypoints[:closest_index]
        waypoints.append([waypoints[0][0], waypoints[0][1], obj.name])
        
        return waypoints

    def generate_exploration_waypoints(self) -> List[List[float]]:
        """Generate waypoints for exploring all objects sequentially and returning to start."""
        all_waypoints = []
        
        for obj_state in self.object_states:
            # Skip objects that shouldn't be scanned
            if not obj_state.obj.should_scan:
                print(f"Skipping waypoint generation for {obj_state.obj.name}")
                continue
            
            if not obj_state.scanned:
                obj_waypoints = self.generate_object_exploration_waypoints(obj_state.obj)
                all_waypoints.extend(obj_waypoints)
                print(f"Generated waypoints for {obj_state.obj.name}")
        
        if RETURN_TO_START and all_waypoints:
            last_point = all_waypoints[-1]
            intermediate = [
                (last_point[0] + self.start_position[0]) / 2,
                (last_point[1] + self.start_position[1]) / 2,
                "return"
            ]
            all_waypoints.append(intermediate)
            all_waypoints.append([self.start_position[0], self.start_position[1], "start"])
        
        return all_waypoints

    def replace_inf_with_max(self, scan, max_distance=4.0):
        scan = np.array(scan)
        scan[~np.isfinite(scan)] = max_distance
        return scan.tolist()

    def run(self):
        while self.robot.step(self.timestep) != -1:
            scan = self.lidar.getRangeImage()
            # Replace inf with 4.0 before logging
            scan = self.replace_inf_with_max(scan, 4.0)
            # Log raw data for YOLO 1D training
            log_lidar_data(self.frame_id, scan, self.angles, self.pose, self.log_file)
            self.frame_id += 1  # Increment frame counter
            # Choose movement mode
            if movement_mode == 0:
                self.move_braitenberg(scan)
            else:
                self.move_to_waypoint()

if __name__ == "__main__":
    controller = E2EController()
    controller.run() 