from controller import Robot, Motor, Lidar, PositionSensor, Supervisor
import math
import json
from typing import List, Dict, NamedTuple, Tuple
import numpy as np
import os
from datetime import datetime

# Global configuration
movement_mode = 1

# Constants
TIME_STEP = 64
EPSILON = 0.25  # Distance discontinuity threshold
BASE_SPEED = 2.0
CIRCLE_RADIUS = 1.2  # meters
WAYPOINT_DISTANCE_THRESHOLD = 0.1  # meters
TURN_GAIN = 1.0
MERGE_WRAP_AROUND = False  # Whether to merge clusters that wrap around the scan

# Navigation constants
SAFETY_MARGIN = 0.35  # 20cm additional distance from box boundary
MIN_WAYPOINT_SPACING = 0.2  # Minimum distance between waypoints
SCAN_DISTANCE = 0.0  # No need for additional scan distance since we're already calculating from center

# New control options
ADAPTIVE_SPEED = False  # Set to True for speed adaptation, False for constant speed
RETURN_TO_START = True  # Whether to return to starting point after scanning all objects

# Obstacle avoidance parameters
OBSTACLE_THRESHOLD = 0.4  # Distance to start avoiding obstacles
AVOIDANCE_WEIGHT = 1.5    # Weight of obstacle avoidance vs goal-seeking
MIN_FRONT_DIST = 0.3      # Minimum front clearance

# Define Object2D class
class Object2D(NamedTuple):
    """Represents a 2D object in the environment."""
    x: float  # center x position
    y: float  # center y position
    width: float  # width of object
    depth: float  # depth of object
    name: str  # identifier

# Define known objects in the environment
KNOWN_OBJECTS = [
    Object2D(x=-0.88499, y=0.982418, width=0.4, depth=0.4, name="Box1"),
    Object2D(x=0.747089, y=-1.05883, width=0.4, depth=0.4, name="Box2")
]

def gaussian(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma))

def create_circular_waypoints(center_x: float, center_y: float, radius: float, num_points: int = 8, object_name: str = None) -> List[List[float]]:
    """Create waypoints in a circle around the center point."""
    waypoints = []
    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        waypoints.append([x, y, object_name])
    return waypoints

def polar_to_cartesian(r, theta):
    """Convert polar coordinates to Cartesian coordinates."""
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return [x, y]

def compute_cluster_centroid(xy_points):
    """Compute the centroid of a cluster given its Cartesian points."""
    if not xy_points:
        return [float('nan'), float('nan')]
    
    x_coords = [p[0] for p in xy_points]
    y_coords = [p[1] for p in xy_points]
    return [sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)]

def compute_geometric_features(cluster):
    """Compute geometric features for a cluster."""
    # Convert input to numpy arrays and handle invalid points
    points = np.array(cluster['points'], dtype=float)
    angles = np.array(cluster['angles'], dtype=float)
    valid_mask = ~np.isnan(points) & ~np.isinf(points)
    
    # Early return for invalid clusters
    if np.sum(valid_mask) < 2:
        return {
            'xy_points': [],
            'centroid': [float('nan'), float('nan')],
            'line_start': [float('nan'), float('nan')],
            'line_end': [float('nan'), float('nan')],
            'line_length': float('nan'),
            'line_angle': float('nan'),
            'opening_angle': float('nan'),
            'perimeter': float('nan'),
            'mean_spacing': float('nan'),
            'bounding_box': [float('nan')] * 4,
            'symmetry_score': float('nan'),
            'corners': [],
            'corner_angles': [],
            'corner_labels': []
        }
    
    # Extract valid points and convert to Cartesian coordinates
    valid_points = points[valid_mask]
    valid_angles = angles[valid_mask]
    x = valid_points * np.cos(valid_angles)
    y = valid_points * np.sin(valid_angles)
    
    # Convert to Python list for JSON serialization
    xy_points = np.column_stack((x, y)).tolist()
    
    # Compute basic features
    centroid = [float(np.mean(x)), float(np.mean(y))]
    line_start = [float(x[0]), float(y[0])]
    line_end = [float(x[-1]), float(y[-1])]
    
    # Compute line features
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    line_length = float(np.sqrt(dx*dx + dy*dy))
    line_angle = float(np.arctan2(dy, dx))
    
    # Compute angular features
    opening_angle = float(valid_angles[-1] - valid_angles[0])
    
    # Compute perimeter and spacing
    dx_diff = np.diff(x)
    dy_diff = np.diff(y)
    distances = np.sqrt(dx_diff*dx_diff + dy_diff*dy_diff)
    perimeter = float(np.sum(distances))
    mean_spacing = float(np.mean(distances)) if len(distances) > 0 else 0.0
    
    # Compute bounding box
    bounding_box = [
        float(np.min(x)), float(np.max(x)),
        float(np.min(y)), float(np.max(y))
    ]
    
    # Compute symmetry score
    symmetry_score = 0.0
    if len(xy_points) > 2:
        # Get the line equation (ax + by + c = 0)
        a = -dy
        b = dx
        c = -(a * float(x[0]) + b * float(y[0]))
        
        # Normalize line equation
        norm = float(np.sqrt(a*a + b*b))
        if norm > 0:
            a /= norm
            b /= norm
            c /= norm
            
            # Project points onto line and compute reflection
            projected = []
            for point in xy_points:
                # Distance from point to line
                d = a * point[0] + b * point[1] + c
                # Reflected point
                reflected = [
                    point[0] - 2 * a * d,
                    point[1] - 2 * b * d
                ]
                projected.append(reflected)
            
            # Compute average distance between points and their reflections
            distances = []
            for i, point in enumerate(xy_points):
                min_dist = float('inf')
                for j, ref in enumerate(projected):
                    if i != j:
                        dist = np.sqrt((point[0] - ref[0])**2 + (point[1] - ref[1])**2)
                        min_dist = min(min_dist, float(dist))
                distances.append(min_dist)
            
            # Convert distances to symmetry score (closer to 0 is more symmetric)
            max_dist = line_length * 0.1  # 10% of line length as threshold
            symmetry_scores = [1 - min(d/max_dist, 1) for d in distances]
            symmetry_score = float(np.mean(symmetry_scores))
    
    # Compute corners using Douglas-Peucker algorithm
    def douglas_peucker(points, epsilon):
        if len(points) < 3:
            return points
        
        # Find point with maximum distance
        dmax = 0
        index = 0
        end = len(points) - 1
        
        for i in range(1, end):
            d = perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d
        
        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            rec_results1 = douglas_peucker(points[:index+1], epsilon)
            rec_results2 = douglas_peucker(points[index:], epsilon)
            
            # Build the result list
            result = rec_results1[:-1] + rec_results2
        else:
            result = [points[0], points[end]]
        
        return result
    
    def perpendicular_distance(point, line_start, line_end):
        # Calculate the perpendicular distance from point to line
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
        numerator = abs((y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x1)
        denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        return numerator/denominator if denominator != 0 else 0
    
    # Convert points to numpy array for processing
    points_array = np.array(xy_points)
    
    # Use Douglas-Peucker algorithm to find corners
    epsilon = line_length * 0.05  # 5% of line length as threshold
    corner_points = douglas_peucker(points_array.tolist(), epsilon)
    
    # Convert corner points to list format and compute corner angles
    corners = []
    corner_angles = []
    corner_labels = []
    
    # Corner detection parameters
    MAX_CORNER_ANGLE = 160.0  # degrees
    MIN_CORNER_DISTANCE = 0.05  # meters
    MAX_CORNERS = 4
    
    if len(corner_points) > 2:
        last_corner = None
        for i in range(1, len(corner_points)-1):
            # Get three consecutive points
            p1 = corner_points[i-1]
            p2 = corner_points[i]
            p3 = corner_points[i+1]
            
            # Compute vectors
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)
            
            # Compute angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            
            # Check if this is a sharp enough corner
            if angle_deg < MAX_CORNER_ANGLE:
                # Check distance from previous corner
                if last_corner is None:
                    # First corner, always add it
                    corners.append(p2)
                    corner_angles.append(angle_deg)
                    corner_labels.append('sharp' if angle_deg < 60.0 else 'shallow')
                    last_corner = p2
                else:
                    # Check distance from previous corner
                    dist = np.sqrt((p2[0] - last_corner[0])**2 + (p2[1] - last_corner[1])**2)
                    if dist >= MIN_CORNER_DISTANCE:
                        corners.append(p2)
                        corner_angles.append(angle_deg)
                        corner_labels.append('sharp' if angle_deg < 60.0 else 'shallow')
                        last_corner = p2
                        
                        # Check if we've reached the maximum number of corners
                        if len(corners) >= MAX_CORNERS:
                            break
    
    # Return all features as a dictionary
    return {
        'xy_points': xy_points,
        'centroid': centroid,
        'line_start': line_start,
        'line_end': line_end,
        'line_length': line_length,
        'line_angle': line_angle,
        'opening_angle': opening_angle,
        'perimeter': perimeter,
        'mean_spacing': mean_spacing,
        'bounding_box': bounding_box,
        'symmetry_score': symmetry_score,
        'corners': corners,
        'corner_angles': corner_angles,
        'corner_labels': corner_labels
    }

def segment_lidar_fagundes(scan: List[float], angles: List[float], epsilon: float = EPSILON, merge_wrap_around: bool = MERGE_WRAP_AROUND) -> List[Dict]:
    """
    Segment LiDAR scan based on distance and angle discontinuities.
    
    Args:
        scan: List of distance measurements
        angles: List of corresponding angles
        epsilon: Threshold for distance discontinuity
        merge_wrap_around: Whether to merge clusters that wrap around the scan
        
    Returns:
        List of clusters, each containing geometric features
    """
    # Constants for segmentation
    MIN_POINTS = 6  # Minimum points in a cluster
    MIN_LINE_LENGTH = 0.2  # meters
    ANGLE_THRESHOLD = math.radians(15)  # 15 degrees in radians
    SHARP_CORNER_THRESHOLD = 60.0  # degrees
    
    clusters = []
    current_cluster = {
        "start_index": 0,
        "end_index": 0,
        "size": 0,
        "angles": [],
        "points": []
    }
    
    def compute_angle_change(p1, p2, p3):
        """Compute the angle change between three consecutive points."""
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return math.pi  # Maximum angle change
            
        # Compute angle between vectors
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return math.acos(cos_angle)
    
    def should_merge_clusters(cluster1, cluster2):
        """Check if two clusters should be merged based on distance and angle continuity."""
        if not cluster1["points"] or not cluster2["points"]:
            return False
            
        # Get the last point of first cluster and first point of second cluster
        p1 = cluster1["points"][-1]
        p2 = cluster2["points"][0]
        
        # Check distance continuity
        if abs(p2 - p1) > epsilon:
            return False
            
        # Check angle continuity if we have enough points
        if len(cluster1["points"]) >= 2 and len(cluster2["points"]) >= 2:
            # Get three points around the potential merge point
            p_prev = cluster1["points"][-2]
            p_curr = p1
            p_next = cluster2["points"][1]
            
            # Convert to Cartesian coordinates
            theta_prev = cluster1["angles"][-2]
            theta_curr = cluster1["angles"][-1]
            theta_next = cluster2["angles"][1]
            
            prev_xy = [p_prev * math.cos(theta_prev), p_prev * math.sin(theta_prev)]
            curr_xy = [p_curr * math.cos(theta_curr), p_curr * math.sin(theta_curr)]
            next_xy = [p_next * math.cos(theta_next), p_next * math.sin(theta_next)]
            
            # Compute angle change
            angle_change = compute_angle_change(prev_xy, curr_xy, next_xy)
            if angle_change > ANGLE_THRESHOLD:
                return False
                
        return True
    
    # First pass: segment based on distance and angle discontinuities
    for i in range(len(scan) - 1):
        # Skip invalid points
        if math.isinf(scan[i]) or math.isnan(scan[i]):
            continue
            
        # Check for discontinuity
        if abs(scan[i+1] - scan[i]) > epsilon:
            # Save current cluster if it has points
            if current_cluster["size"] > 0:
                clusters.append(current_cluster)
            
            # Start new cluster
            current_cluster = {
                "start_index": i + 1,
                "end_index": i + 1,
                "size": 0,
                "angles": [],
                "points": []
            }
        else:
            # Add point to current cluster
            current_cluster["end_index"] = i + 1
            current_cluster["size"] += 1
            current_cluster["angles"].append(angles[i])
            current_cluster["points"].append(scan[i])
    
    # Add last cluster if it has points
    if current_cluster["size"] > 0:
        clusters.append(current_cluster)
    
    # Second pass: merge clusters if needed
    if len(clusters) >= 2:
        merged_clusters = []
        i = 0
        
        while i < len(clusters):
            current = clusters[i]
            
            # Check if we should merge with next cluster
            if i < len(clusters) - 1 and should_merge_clusters(current, clusters[i+1]):
                # Merge clusters
                merged = {
                    "start_index": current["start_index"],
                    "end_index": clusters[i+1]["end_index"],
                    "size": current["size"] + clusters[i+1]["size"],
                    "angles": current["angles"] + clusters[i+1]["angles"],
                    "points": current["points"] + clusters[i+1]["points"]
                }
                merged_clusters.append(merged)
                i += 2  # Skip the next cluster as it's been merged
            else:
                merged_clusters.append(current)
                i += 1
        
        clusters = merged_clusters
    
    # Check for wrap-around segmentation if enabled
    if merge_wrap_around and len(clusters) >= 2:
        first_cluster = clusters[0]
        last_cluster = clusters[-1]
        
        if should_merge_clusters(last_cluster, first_cluster):
            # Merge the clusters
            merged_cluster = {
                "start_index": last_cluster["start_index"],
                "end_index": first_cluster["end_index"],
                "size": last_cluster["size"] + first_cluster["size"],
                "angles": last_cluster["angles"] + first_cluster["angles"],
                "points": last_cluster["points"] + first_cluster["points"]
            }
            
            # Remove the first and last clusters and add the merged one
            clusters = clusters[1:-1] + [merged_cluster]
    
    # Third pass: compute features and filter noise clusters
    filtered_clusters = []
    for cluster in clusters:
        # Compute geometric features
        features = compute_geometric_features(cluster)
        cluster.update(features)
        
        # Check if cluster should be kept
        keep_cluster = True
        
        # Check size and length
        if cluster["size"] < MIN_POINTS and cluster["line_length"] < MIN_LINE_LENGTH:
            # Check for sharp corners
            has_sharp_corner = False
            for angle in cluster.get("corner_angles", []):
                if angle < SHARP_CORNER_THRESHOLD:
                    has_sharp_corner = True
                    break
            
            if not has_sharp_corner:
                keep_cluster = False
        
        if keep_cluster:
            filtered_clusters.append(cluster)
    
    # Add cluster IDs
    for i, cluster in enumerate(filtered_clusters):
        cluster["cluster_id"] = i
    
    return filtered_clusters

def log_lidar_data(timestamp: float, raw_scan: List[float], angles: List[float], 
                  clusters: List[Dict], pose: List[float], filename: str):
    """
    Log LiDAR data to JSON Lines file.
    
    Args:
        timestamp: Current simulation time
        raw_scan: Raw LiDAR scan data
        angles: Corresponding angles
        clusters: List of segmented clusters with geometric features
        pose: Robot pose [x, y, theta]
        filename: Output file name
    """
    log_entry = {
        "timestamp": timestamp,
        "raw_scan": raw_scan,
        "angles": angles,
        "pose": pose,
        "clusters": clusters,
        "num_clusters": len(clusters)
    }
    
    with open(filename, 'a') as f:
        json.dump(log_entry, f)
        f.write('\n')

def generate_box_perimeter(obj: Object2D, num_points: int = 8) -> List[List[float]]:
    """Generate waypoints around a rectangular object."""
    # Calculate box dimensions with safety margin
    half_width = (obj.width / 2) + SAFETY_MARGIN + SCAN_DISTANCE
    half_depth = (obj.depth / 2) + SAFETY_MARGIN + SCAN_DISTANCE
    
    # Generate corner points
    corners = [
        [obj.x + half_width, obj.y + half_depth],   # Top-right
        [obj.x - half_width, obj.y + half_depth],   # Top-left
        [obj.x - half_width, obj.y - half_depth],   # Bottom-left
        [obj.x + half_width, obj.y - half_depth]    # Bottom-right
    ]
    
    waypoints = []
    # Interpolate points between corners
    points_per_side = max(2, num_points // 4)
    
    for i in range(4):
        start = corners[i]
        end = corners[(i + 1) % 4]
        
        # Generate points along this side
        for t in range(points_per_side):
            alpha = t / points_per_side
            point = [
                start[0] + (end[0] - start[0]) * alpha,
                start[1] + (end[1] - start[1]) * alpha
            ]
            waypoints.append(point)
    
    return waypoints

def generate_circular_perimeter(obj: Object2D, num_points: int = 16) -> List[List[float]]:
    """Generate waypoints in a circle around an object."""
    # Use the larger dimension plus safety margin as radius
    radius = max(obj.width, obj.depth) / 2 + SAFETY_MARGIN + SCAN_DISTANCE
    
    waypoints = []
    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points
        x = obj.x + radius * math.cos(angle)
        y = obj.y + radius * math.sin(angle)
        waypoints.append([x, y])
    
    return waypoints

def optimize_waypoints(waypoints: List[List[float]]) -> List[List[float]]:
    """Optimize waypoint sequence and ensure minimum spacing."""
    if not waypoints:
        return []
    
    optimized = [waypoints[0]]
    
    for point in waypoints[1:]:
        last_point = optimized[-1]
        # Check distance from last point
        dist = math.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
        
        if dist >= MIN_WAYPOINT_SPACING:
            optimized.append(point)
    
    return optimized

class ObjectState:
    def __init__(self, obj: Object2D):
        self.obj = obj
        self.scanned = False
        self.last_scan_time = 0
        self.scan_quality = 0  # 0-1 score of how well we've scanned it
        self.scan_points = []  # List of points where we scanned this object

class TurtleBot3Controller:
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
        self.angles = [i * self.angle_increment for i in range(self.lidar_width)]
        
        # Initialize Braitenberg coefficients
        self.braitenberg_coefficients = []
        for i in range(self.lidar_width):
            self.braitenberg_coefficients.append(6 * gaussian(i, self.lidar_width / 4, self.lidar_width / 12))
        
        # Add flag for completion message
        self.completion_message_shown = False
        self.mission_complete = False  # New flag to track mission completion
        
        # Initialize object states
        self.object_states = [ObjectState(obj) for obj in KNOWN_OBJECTS]
        self.start_position = [0.0, 0.0]  # Store starting position
        
        # Initialize pose tracking
        self.pose = [0.0, 0.0, 0.0]  # x, y, theta
        
        # Generate initial waypoints
        self.waypoints = self.generate_exploration_waypoints()
        self.current_waypoint_index = 0
        
        # Initialize logging
        self.log_file = self._init_logging()
        print(f"[Logger] Logging to: {self.log_file}")

    def _init_logging(self):
        """Initialize logging directory and create log file."""
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, 'logs')
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f'scan_log_{timestamp}.jsonl'
        
        # Return full path to log file
        return os.path.join(logs_dir, log_filename)

    def update_pose(self):
        """Update robot pose using supervisor position and orientation."""
        # Get position from supervisor
        position = self.robot_node.getPosition()
        rotation = self.robot_node.getOrientation()
        
        # Update x, y position
        self.pose[0] = position[0]
        self.pose[1] = position[1]
        
        # Calculate theta from rotation matrix
        # The rotation matrix is in row-major order
        # For a rotation around the Z-axis, we can get theta from atan2(R[3], R[0])
        self.pose[2] = math.atan2(rotation[3], rotation[0])
        
        # Normalize theta to [-pi, pi]
        while self.pose[2] > math.pi:
            self.pose[2] -= 2 * math.pi
        while self.pose[2] < -math.pi:
            self.pose[2] += 2 * math.pi

    def move_braitenberg(self, lidar_values):
        """Move using Braitenberg coefficients."""
        left_speed = BASE_SPEED
        right_speed = BASE_SPEED
        
        for i in range(int(0.25 * self.lidar_width), int(0.5 * self.lidar_width)):
            j = self.lidar_width - i - 1
            k = i - int(0.25 * self.lidar_width)
            
            if (not math.isinf(lidar_values[i]) and not math.isnan(lidar_values[i]) and 
                not math.isinf(lidar_values[j]) and not math.isnan(lidar_values[j])):
                
                left_speed += self.braitenberg_coefficients[k] * (
                    (1.0 - lidar_values[i] / self.lidar.getMaxRange()) - 
                    (1.0 - lidar_values[j] / self.lidar.getMaxRange())
                )
                
                right_speed += self.braitenberg_coefficients[k] * (
                    (1.0 - lidar_values[j] / self.lidar.getMaxRange()) - 
                    (1.0 - lidar_values[i] / self.lidar.getMaxRange())
                )
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def update_object_states(self, clusters: List[Dict]):
        """Update the state of known objects based on current scan."""
        total_waypoints_per_object = 9  # 8 circle points + 1 return point
        total_objects = len(self.object_states)
        total_object_waypoints = total_objects * total_waypoints_per_object
        
        # If we're in the return phase, don't update object states
        if self.current_waypoint_index >= total_object_waypoints:
            return
            
        # Calculate which object we're currently scanning
        current_obj_index = self.current_waypoint_index // total_waypoints_per_object
        current_waypoint = (self.current_waypoint_index % total_waypoints_per_object) + 1
        
        # Only process if we're scanning a valid object
        if current_obj_index < total_objects:
            obj_state = self.object_states[current_obj_index]
            
            # Check if we've completed a full circle
            if current_waypoint == total_waypoints_per_object and not obj_state.scanned:
                # Mark this object as scanned
                obj_state.scanned = True
                obj_state.last_scan_time = self.robot.getTime()
                obj_state.scan_quality = 1.0
                obj_state.scan_points.append([self.pose[0], self.pose[1]])
                print(f"Object {obj_state.obj.name} scanned after completing full circle")
            
            # Verify the object is still visible
            coverage = self._calculate_object_coverage(obj_state.obj, clusters)
            if coverage < 0.5:  # If we lose sight of a scanned object
                obj_state.scanned = False  # Reset if we lose sight
        
        # Check if all objects are scanned
        all_scanned = all(obj.scanned for obj in self.object_states)
        
        # Only show completion message once when all objects are actually scanned
        if all_scanned and not self.completion_message_shown:
            print("All objects scanned! Returning to start position...")
            self.completion_message_shown = True

    def move_to_waypoint(self):
        """Enhanced waypoint navigation with object state tracking."""
        # If mission is complete, stay stopped
        if self.mission_complete:
            self.left_motor.setVelocity(0)
            self.right_motor.setVelocity(0)
            return
            
        # Update robot pose using supervisor
        self.update_pose()
        
        # Get current waypoint
        target_x, target_y, waypoint_owner = self.waypoints[self.current_waypoint_index]
        
        # Calculate distance and angle to waypoint
        dx = target_x - self.pose[0]
        dy = target_y - self.pose[1]
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.atan2(dy, dx)
        
        # Get LiDAR scan
        scan = self.lidar.getRangeImage()
        clusters = segment_lidar_fagundes(scan, self.angles)
        
        # Update object states
        self.update_object_states(clusters)
        
        # Check if we've reached the waypoint
        if distance < WAYPOINT_DISTANCE_THRESHOLD:
            # Calculate current object index and waypoint number
            total_waypoints_per_object = 9  # 8 circle points + 1 return point
            total_objects = len(self.object_states)
            total_object_waypoints = total_objects * total_waypoints_per_object
            
            # Check if we're in the return-to-start phase
            if waypoint_owner == "return" or waypoint_owner == "start":
                # If we've reached the start position (with some tolerance)
                if waypoint_owner == "start":
                    # Check if we're close enough to the start position
                    dx_to_start = self.pose[0] - self.start_position[0]
                    dy_to_start = self.pose[1] - self.start_position[1]
                    distance_to_start = math.sqrt(dx_to_start*dx_to_start + dy_to_start*dy_to_start)
                    
                    if distance_to_start < 0.1:  # 10cm tolerance for final position
                        print("Mission complete! All objects scanned and returned to start position.")
                        print(f"Final position: ({self.pose[0]:.3f}, {self.pose[1]:.3f})")
                        self.left_motor.setVelocity(0)
                        self.right_motor.setVelocity(0)
                        self.mission_complete = True  # Set mission complete flag
                        return
                    else:
                        print(f"Approaching start position... Current distance: {distance_to_start:.3f}m")
                        # If we're very close but not quite there, just stop
                        if distance_to_start < 0.2:  # 20cm threshold for stopping
                            print("Close enough to start position, mission complete!")
                            self.left_motor.setVelocity(0)
                            self.right_motor.setVelocity(0)
                            self.mission_complete = True
                            return
                else:
                    print("Returning to start position...")
            else:
                # Find which object we're currently scanning
                current_obj = next((obj for obj in self.object_states if obj.obj.name == waypoint_owner), None)
                if current_obj is not None:
                    # Calculate waypoint number within this object's sequence
                    obj_waypoints = [w for w in self.waypoints if w[2] == waypoint_owner]
                    waypoint_in_circle = obj_waypoints.index(self.waypoints[self.current_waypoint_index]) + 1
                    
                    # If this is the first waypoint of a new object
                    if waypoint_in_circle == 1:
                        print(f"\nStarting exploration around {current_obj.obj.name} at ({current_obj.obj.x:.2f}, {current_obj.obj.y:.2f})")
                        print(f"Total waypoints for this object: {total_waypoints_per_object} (8 circle points + 1 return point)")
                    
                    # Log each waypoint reached with correct numbering
                    print(f"Reached waypoint {waypoint_in_circle}/{total_waypoints_per_object} for object {current_obj.obj.name}")
            
            # Move to next waypoint only if mission is not complete
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
        if ADAPTIVE_SPEED:
            base_speed = min(BASE_SPEED, distance * 2.0)
        else:
            base_speed = BASE_SPEED
        
        # Progressive speed reduction when approaching waypoint
        if distance < 0.3:  # Start slowing down at 30cm
            # Gradually reduce speed as we get closer
            # At 30cm: full speed, at 0cm: 20% speed
            speed_factor = max(0.2, distance / 0.3)
            base_speed *= speed_factor
            
        # Adjust speed based on angle error
        if abs(angular_error) > math.pi/4:  # If we need to turn more than 45 degrees
            base_speed *= 0.5  # Slow down for sharp turns
            
        left_speed = base_speed - TURN_GAIN * angular_error
        right_speed = base_speed + TURN_GAIN * angular_error
        
        # Apply obstacle avoidance if needed
        if self._check_obstacle_proximity(clusters):
            left_speed, right_speed = self._avoid_obstacle(clusters, left_speed, right_speed)
            
            # Emergency stop if too close to obstacle
            if self._get_min_obstacle_distance(clusters) < MIN_FRONT_DIST:
                left_speed = -BASE_SPEED * 0.5
                right_speed = -BASE_SPEED * 0.5
        
        # Set motor speeds
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def _calculate_object_coverage(self, obj: Object2D, clusters: List[Dict]) -> float:
        """Calculate how well an object is covered by the current scan."""
        # Convert object to polar coordinates relative to robot
        dx = obj.x - self.pose[0]
        dy = obj.y - self.pose[1]
        obj_distance = math.sqrt(dx*dx + dy*dy)
        obj_angle = math.atan2(dy, dx)
        
        # Find clusters that might be this object
        relevant_clusters = []
        for cluster in clusters:
            # Check if cluster is in the general direction of the object
            cluster_angle = math.atan2(cluster['centroid'][1], cluster['centroid'][0])
            angle_diff = abs(cluster_angle - obj_angle)
            if angle_diff < math.pi/4:  # Within 45 degrees
                relevant_clusters.append(cluster)
        
        if not relevant_clusters:
            return 0.0
        
        # Calculate coverage based on cluster features
        total_coverage = 0.0
        for cluster in relevant_clusters:
            # Check if cluster size matches object size
            cluster_size = cluster.get('line_length', 0)
            expected_size = max(obj.width, obj.depth)
            size_match = min(cluster_size / expected_size, 1.0)
            
            # Check if distance is reasonable
            cluster_dist = math.sqrt(cluster['centroid'][0]**2 + cluster['centroid'][1]**2)
            dist_match = 1.0 - min(abs(cluster_dist - obj_distance) / obj_distance, 1.0)
            
            # Check if we have good point coverage
            point_coverage = len(cluster.get('points', [])) / (self.lidar_width / 4)  # Expect at least 1/4 of points
            point_coverage = min(point_coverage, 1.0)
            
            # Combine factors with weights
            coverage = (size_match * 0.4 + dist_match * 0.3 + point_coverage * 0.3)
            total_coverage = max(total_coverage, coverage)
        
        return total_coverage

    def _check_obstacle_proximity(self, clusters: List[Dict]) -> bool:
        """Check if there are obstacles too close to the robot."""
        for cluster in clusters:
            # Check distance to cluster centroid
            dx = cluster['centroid'][0] - self.pose[0]
            dy = cluster['centroid'][1] - self.pose[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < OBSTACLE_THRESHOLD:
                return True
        return False

    def _avoid_obstacle(self, clusters: List[Dict], left_speed: float, right_speed: float) -> Tuple[float, float]:
        """Adjust speeds to avoid obstacles."""
        # Find the closest obstacle
        closest_dist = float('inf')
        closest_angle = 0.0
        
        for cluster in clusters:
            dx = cluster['centroid'][0] - self.pose[0]
            dy = cluster['centroid'][1] - self.pose[1]
            distance = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx)
            
            if distance < closest_dist:
                closest_dist = distance
                closest_angle = angle
        
        # Calculate avoidance factor
        avoidance_factor = (OBSTACLE_THRESHOLD - closest_dist) / OBSTACLE_THRESHOLD
        
        # Adjust speeds based on obstacle position
        if closest_angle > 0:  # Obstacle on the left
            left_speed *= (1.0 - avoidance_factor * AVOIDANCE_WEIGHT)
            right_speed *= (1.0 + avoidance_factor * AVOIDANCE_WEIGHT)
        else:  # Obstacle on the right
            left_speed *= (1.0 + avoidance_factor * AVOIDANCE_WEIGHT)
            right_speed *= (1.0 - avoidance_factor * AVOIDANCE_WEIGHT)
        
        return left_speed, right_speed

    def generate_object_exploration_waypoints(self, obj: Object2D) -> List[List[float]]:
        """Generate waypoints in a circle around the object."""
        # Calculate radius: half of box size + safety margin
        radius = (max(obj.width, obj.depth) / 2) + SAFETY_MARGIN
        
        # Generate all points in a circle
        circle_points = create_circular_waypoints(obj.x, obj.y, radius, 8, obj.name)
        
        # Find the closest waypoint to robot's current position
        min_dist = float('inf')
        closest_index = 0
        for i, point in enumerate(circle_points):
            dx = point[0] - self.pose[0]
            dy = point[1] - self.pose[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        
        # Reorder waypoints to start from closest point
        waypoints = circle_points[closest_index:] + circle_points[:closest_index]
        
        # Add the first point again to complete the circle
        waypoints.append([waypoints[0][0], waypoints[0][1], obj.name])
        
        return waypoints

    def generate_exploration_waypoints(self) -> List[List[float]]:
        """Generate waypoints for exploring all objects sequentially and returning to start."""
        all_waypoints = []
        current_position = [self.pose[0], self.pose[1]]  # Start from current position
        
        # Scan objects in fixed order (Box1 then Box2)
        for obj_state in self.object_states:
            if not obj_state.scanned:
                # Generate waypoints for this object
                obj_waypoints = self.generate_object_exploration_waypoints(obj_state.obj)
                
                # Add object waypoints
                all_waypoints.extend(obj_waypoints)
                
                # Don't mark as scanned here - wait for complete circle
                print(f"Generated waypoints for {obj_state.obj.name}")
        
        # Add return to start if enabled
        if RETURN_TO_START and all_waypoints:
            last_point = all_waypoints[-1]
            # Add intermediate point for smoother return
            intermediate = [
                (last_point[0] + self.start_position[0]) / 2,
                (last_point[1] + self.start_position[1]) / 2,
                "return"  # Mark as return waypoint
            ]
            all_waypoints.append(intermediate)
            all_waypoints.append([self.start_position[0], self.start_position[1], "start"])
        
        # Verify all waypoints have the correct format
        for i, waypoint in enumerate(all_waypoints):
            if len(waypoint) != 3:
                print(f"Error: Waypoint {i} has incorrect format: {waypoint}")
                raise ValueError(f"Waypoint {i} must have format [x, y, owner]")
        
        return all_waypoints

    def _get_min_obstacle_distance(self, clusters: List[Dict]) -> float:
        """Get the minimum distance to any obstacle."""
        min_dist = float('inf')
        for cluster in clusters:
            dx = cluster['centroid'][0] - self.pose[0]
            dy = cluster['centroid'][1] - self.pose[1]
            distance = math.sqrt(dx*dx + dy*dy)
            min_dist = min(min_dist, distance)
        return min_dist

    def run(self):
        while self.robot.step(self.timestep) != -1:
            # Get current time
            timestamp = self.robot.getTime()
            
            # Get LiDAR scan
            scan = self.lidar.getRangeImage()
            
            # Segment scan
            clusters = segment_lidar_fagundes(scan, self.angles, merge_wrap_around=MERGE_WRAP_AROUND)
            
            # Log data
            log_lidar_data(timestamp, scan, self.angles, clusters, self.pose, self.log_file)
            
            # Choose movement mode
            if movement_mode == 0:
                self.move_braitenberg(scan)
            else:
                self.move_to_waypoint()

if __name__ == "__main__":
    controller = TurtleBot3Controller()
    controller.run() 