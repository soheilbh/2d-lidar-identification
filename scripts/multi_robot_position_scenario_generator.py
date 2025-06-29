# Multi-Robot Position Scenario Generator
# This script generates multiple scenarios with randomized object placements and robot positions for 2D LiDAR simulation environments.
# It ensures objects do not overlap, robot positions are valid, and waypoints are generated for navigation tasks.
# The script supports visualization and exports scenario data for downstream tasks.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms
import math
import random
import os
import json
import datetime
from tqdm import tqdm
import sys
import heapq
from collections import OrderedDict

# Add project root to sys.path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from bot3.utils.lidar_utils import KNOWN_OBJECTS

# --- Configuration Parameters ---
NUM_SCENARIOS = 2  # Number of scenarios to generate
NUM_ROBOT_POSITION = 50  # Number of robot positions per scenario
NUM_WAYPOINTS_PER_POSITION = 2  # Number of waypoints to generate for each robot position
ROBOT_POSITION_CLEARANCE = 0.05   # Minimum clearance for robot positions (meters)
SHOW_MAIN_VISUALIZATION = False # Set to True to show and save the main visualization with all scenarios
WAYPOINTS_RELATIVE_TO_HEADING = True  # Generate waypoints relative to robot heading

# Object sizes (width, height) in meters
SIZES = {
    'box': (0.4, 0.6),
    'chair': (0.4, 0.4),
    'desk': (0.8, 1.6)
}
MIN_CLEARANCE = 0.7  # Minimum clearance between objects (meters)

WAYPOINT_EXCLUSION_RADIUS = 0.6  # meters, exclusion zone for waypoints near objects
ASTAR_CLEARANCE = 0.25  # meters, used for A* path planning

class FloatEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to round floats for compact scenario export.
    Handles numpy floats, lists, and dicts recursively.
    """
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64, float)):
            return round(float(obj), 2)
        elif isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.default(value) for key, value in obj.items()}
        return super().default(obj)

# --- Helper Functions ---

def get_randomized_size(obj_name):
    """
    Get a randomized size for an object based on its base size and random scaling.
    Args:
        obj_name (str): Name of the object ('box', 'chair', 'desk')
    Returns:
        tuple: (width, height) in meters
    """
    base_w, base_h = SIZES.get(obj_name, (0.4, 0.4))
    if obj_name == 'chair':
        scale = random.uniform(0.8, 1.2)
        return base_w * scale, base_h * scale
    else:
        scale_w = random.uniform(0.8, 1.2)
        scale_h = random.uniform(0.8, 1.2)
        return base_w * scale_w, base_h * scale_h

# Helper for overlap check (Separating Axis Theorem)
def rectangles_overlap(center1, w1, h1, theta1, center2, w2, h2, theta2):
    """
    Check if two rectangles (with rotation) overlap using the Separating Axis Theorem.
    Args:
        center1, w1, h1, theta1: Center, width, height, rotation of first rectangle
        center2, w2, h2, theta2: Center, width, height, rotation of second rectangle
    Returns:
        bool: True if rectangles overlap, False otherwise
    """
    def get_corners(center, w, h, theta):
        dx = w / 2
        dy = h / 2
        corners = np.array([
            [dx, dy],
            [-dx, dy],
            [-dx, -dy],
            [dx, -dy]
        ])
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        return np.dot(corners, rot.T) + center
    def get_axes(corners):
        axes = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            edge = p2 - p1
            # Check if edge has zero length
            edge_length = np.linalg.norm(edge)
            if edge_length > 1e-10:  # Only normalize if edge has significant length
                normal = np.array([-edge[1], edge[0]])
                normal = normal / edge_length
                axes.append(normal)
        return axes
    c1 = np.array(center1)
    c2 = np.array(center2)
    rect1 = get_corners(c1, w1, h1, theta1)
    rect2 = get_corners(c2, w2, h2, theta2)
    axes = get_axes(rect1) + get_axes(rect2)
    for axis in axes:
        proj1 = [np.dot(corner, axis) for corner in rect1]
        proj2 = [np.dot(corner, axis) for corner in rect2]
        if max(proj1) < min(proj2) or max(proj2) < min(proj1):
            return False
    return True

def corners_within_bounds(center, w, h, theta):
    """
    Check if all corners of a rectangle are within the allowed world bounds.
    Args:
        center (array): Center of rectangle
        w, h (float): Width and height
        theta (float): Rotation angle in radians
    Returns:
        bool: True if all corners are within bounds, False otherwise
    """
    dx = w / 2
    dy = h / 2
    corners = np.array([
        [dx, dy],
        [-dx, dy],
        [-dx, -dy],
        [dx, -dy]
    ])
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    world_corners = np.dot(corners, rot.T) + center
    for x, y in world_corners:
        if not (-1.8 <= x <= 1.8 and -1.8 <= y <= 1.8):
            return False
    return True

def is_valid_position(new_pos, rotation, existing_positions, obj_name, obj_size, prev_waypoints=None):
    """
    Check if a new object position is valid (no overlap, within bounds, not too close to waypoints).
    Args:
        new_pos (array): Proposed center position
        rotation (float): Rotation angle in radians
        existing_positions (list): List of existing object positions
        obj_name (str): Name of the object
        obj_size (tuple): (width, height)
        prev_waypoints (list): Previous waypoints to avoid
    Returns:
        bool: True if position is valid, False otherwise
    """
    w1, h1 = obj_size
    w1_clear = w1 + MIN_CLEARANCE
    h1_clear = h1 + MIN_CLEARANCE
    if not corners_within_bounds(np.array(new_pos), w1_clear, h1_clear, rotation):
        return False
    for existing in existing_positions:
        name2 = existing['name']
        w2, h2 = existing['size']
        w2_clear = w2 + MIN_CLEARANCE
        h2_clear = h2 + MIN_CLEARANCE
        center2 = np.array(existing['position'])
        theta2 = existing['rotation']
        if rectangles_overlap(np.array(new_pos), w1_clear, h1_clear, rotation, center2, w2_clear, h2_clear, theta2):
            return False
    if prev_waypoints is not None and len(prev_waypoints) > 0:
        first_wp = np.array(prev_waypoints[0])
        last_wp = np.array(prev_waypoints[-1])
        corners = get_corners_world(np.array(new_pos), w1, h1, rotation)
        for corner in corners:
            if np.linalg.norm(np.array(corner) - first_wp) < WAYPOINT_EXCLUSION_RADIUS or np.linalg.norm(np.array(corner) - last_wp) < WAYPOINT_EXCLUSION_RADIUS:
                return False
    return True

def generate_random_position(existing_positions, obj_name, scenario_index, obj_size=None, prev_waypoints=None):
    """
    Generate a random valid position and rotation for an object, avoiding overlaps and out-of-bounds.
    Args:
        existing_positions (list): List of already placed objects
        obj_name (str): Name of the object
        scenario_index (int): Index of the scenario (for rotation logic)
        obj_size (tuple): (width, height) or None
        prev_waypoints (list): Previous waypoints to avoid
    Returns:
        dict: Position, rotation, name, and size of the object
    """
    max_attempts = 1000
    x_range = (-1.3, 1.3) if obj_name == 'v_box' else (-1.8, 1.8)
    y_range = (-1.3, 1.3) if obj_name == 'v_box' else (-1.8, 1.8)
    if obj_size is None:
        w1, h1 = SIZES.get(obj_name, (0.4, 0.4))
    else:
        w1, h1 = obj_size
    w1_clear = w1 + MIN_CLEARANCE
    h1_clear = h1 + MIN_CLEARANCE
    use_90_degree = scenario_index < (NUM_SCENARIOS // 4)
    for attempt in range(max_attempts):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        if use_90_degree:
            rotation = random.choice([0, math.pi/2, math.pi, 3*math.pi/2])
        else:
            rotation = random.uniform(0, 2 * math.pi)
        valid = True
        if not corners_within_bounds(np.array([x, y]), w1_clear, h1_clear, rotation):
            valid = False
        for existing in existing_positions:
            name2 = existing['name']
            w2, h2 = existing['size']
            w2_clear = w2 + MIN_CLEARANCE
            h2_clear = h2 + MIN_CLEARANCE
            center2 = np.array(existing['position'])
            theta2 = existing['rotation']
            if rectangles_overlap(np.array([x, y]), w1_clear, h1_clear, rotation, center2, w2_clear, h2_clear, theta2):
                valid = False
                break
        if prev_waypoints is not None and len(prev_waypoints) > 0:
            first_wp = np.array(prev_waypoints[0])
            last_wp = np.array(prev_waypoints[-1])
            corners = get_corners_world(np.array([x, y]), w1, h1, rotation)
            for corner in corners:
                if np.linalg.norm(np.array(corner) - first_wp) < WAYPOINT_EXCLUSION_RADIUS or np.linalg.norm(np.array(corner) - last_wp) < WAYPOINT_EXCLUSION_RADIUS:
                    valid = False
                    break
        if valid:
            return {
                'position': [x, y],
                'rotation': rotation,
                'name': obj_name,
                'size': [w1, h1]
            }
    return {
        'position': [0.0, 0.0],
        'rotation': 0.0,
        'name': obj_name,
        'size': [w1, h1]
    }

def generate_scenario_positions(max_order_tries=5, scenario_index=0, prev_waypoints=None):
    """
    Attempt to generate a valid set of positions for all objects in a scenario.
    Randomizes object order and sizes, and tries multiple times to avoid overlaps and out-of-bounds.
    Args:
        max_order_tries (int): Number of order shuffles to try
        scenario_index (int): Index of the scenario (affects rotation logic)
        prev_waypoints (list): Previous waypoints to avoid
    Returns:
        dict: Mapping of object names to their positions, rotations, and sizes, or None if failed
    """
    object_names = ['box', 'chair', 'desk']
    for order_try in range(max_order_tries):
        order = object_names[:]
        random.shuffle(order)
        positions = {}
        existing_positions = []
        failed = False
        # Randomize sizes for this scenario
        obj_sizes = {name: get_randomized_size(name) for name in object_names}
        for obj_name in order:
            new_pos_data = generate_random_position(existing_positions, obj_name, scenario_index, obj_size=obj_sizes[obj_name], prev_waypoints=prev_waypoints)
            if new_pos_data['position'] == [0.0, 0.0] and len(existing_positions) > 0:
                failed = True
                break
            existing_positions.append(new_pos_data)
            positions[obj_name] = {
                'position': new_pos_data['position'],
                'rotation': new_pos_data['rotation'],
                'size': new_pos_data['size']
            }
        if not failed:
            return positions
    return None

def generate_waypoints_for_vbox(vbox):
    """
    Generate waypoints around the corners of a virtual box (v_box).
    Args:
        vbox (dict): Dictionary with 'position' and 'size' keys
    Returns:
        list: List of waypoint coordinates around the box
    """
    width, depth = vbox['size']
    half_width = (width / 2) + 0.1
    half_depth = (depth / 2) + 0.1
    corners = [
        [vbox['position'][0] + half_width, vbox['position'][1] + half_depth],
        [vbox['position'][0] - half_width, vbox['position'][1] + half_depth],
        [vbox['position'][0] - half_width, vbox['position'][1] - half_depth],
        [vbox['position'][0] + half_width, vbox['position'][1] - half_depth]
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
            waypoints.append([point[0], point[1]])
    return waypoints

def add_rotated_rect(ax, center, width, height, angle_rad, color, label, clearance=False, show_center=True, show_label=True, rotation_angle_deg=None):
    """
    Add a rotated rectangle (object or wall) to a matplotlib axis.
    Args:
        ax: Matplotlib axis
        center: Center of rectangle
        width, height: Dimensions
        angle_rad: Rotation angle in radians
        color: Color for the rectangle
        label: Label for legend
        clearance: If True, draw as a clearance zone (transparent)
        show_center: If True, plot the center point
        show_label: If True, show size label
        rotation_angle_deg: Optional, show rotation angle in degrees
    """
    rect = Rectangle((center[0] - width/2, center[1] - height/2), width, height, angle=0, color=color, alpha=0.5 if not clearance else 0.15, label=label if not clearance else None)
    t = mtransforms.Affine2D().rotate_around(center[0], center[1], angle_rad) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    if not clearance:
        if show_center:
            ax.plot(center[0], center[1], 'ko', markersize=3, zorder=10)
        if show_label:
            ax.text(center[0], center[1], f'{width:.2f}×{height:.2f}', color='black', fontsize=7, ha='center', va='center', zorder=11, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
        if rotation_angle_deg is not None:
            ax.text(center[0], center[1]-0.13, f'{rotation_angle_deg:.2f}°', color='black', fontsize=7, ha='center', va='center', zorder=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))

def check_and_highlight_overlaps(ax, positions):
    """
    Check for overlapping objects and highlight them in red on the plot.
    Args:
        ax: Matplotlib axis
        positions: Dictionary of object positions
    """
    # Check all pairs for overlap
    names = list(positions.keys())
    overlaps = set()
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            p1, p2 = positions[n1], positions[n2]
            w1, h1 = p1['size']
            w2, h2 = p2['size']
            if rectangles_overlap(p1['position'], w1, h1, p1['rotation'], p2['position'], w2, h2, p2['rotation']):
                overlaps.add(n1)
                overlaps.add(n2)
    # Highlight overlapping objects in red border
    for n in overlaps:
        p = positions[n]
        w, h = p['size']
        rect = Rectangle((p['position'][0] - w/2, p['position'][1] - h/2), w, h, angle=0, fill=False, edgecolor='red', linewidth=2, zorder=20)
        t = mtransforms.Affine2D().rotate_around(p['position'][0], p['position'][1], p['rotation']) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

def rotate_point(x, y, angle_rad):
    """
    Rotate a point (x, y) by angle_rad radians around the origin.
    Args:
        x, y: Coordinates
        angle_rad: Rotation angle in radians
    Returns:
        list: Rotated coordinates [x', y']
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return [c * x - s * y, s * x + c * y]

def is_point_safe(point, positions, clearance):
    """
    Check if a point is safe (not inside any object's clearance zone).
    Args:
        point: [x, y] coordinates
        positions: Dictionary of object positions
        clearance: Clearance radius
    Returns:
        bool: True if point is safe, False otherwise
    """
    px, py = point
    leg_radius = 0.22  # meters
    for obj_name, obj_data in positions.items():
        obj_pos = np.array(obj_data['position'])
        obj_size = obj_data['size']
        if obj_name in ['chair', 'desk']:
            # Only legs are occupied
            w, h = obj_size
            theta = obj_data['rotation']
            # Get corners (legs) in world coordinates
            dx = w / 2
            dy = h / 2
            corners = np.array([
                [dx, dy],
                [-dx, dy],
                [-dx, -dy],
                [dx, -dy]
            ])
            rot = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            world_corners = np.dot(corners, rot.T) + obj_pos
            for leg in world_corners:
                if np.linalg.norm(np.array([px, py]) - leg) < leg_radius:
                    return False
        else:
            # For box, use shadow box with half clearance
            w, h = obj_size
            theta = obj_data['rotation']
            # Get corners of shadow box (with half clearance)
            dx = (w + MIN_CLEARANCE * 1.8) / 2
            dy = (h + MIN_CLEARANCE * 1.8) / 2
            corners = np.array([
                [dx, dy],
                [-dx, dy],
                [-dx, -dy],
                [dx, -dy]
            ])
            rot = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            world_corners = np.dot(corners, rot.T) + obj_pos
            # Check if point is inside shadow box
            if rectangles_overlap(
                np.array([px, py]), 0.01, 0.01, 0,  # Point as tiny rectangle
                obj_pos, w + MIN_CLEARANCE/2, h + MIN_CLEARANCE/2, theta
            ):
                return False
    return corners_within_bounds(np.array([px, py]), clearance, clearance, 0)

def build_occupancy_grid(positions, clearance=ASTAR_CLEARANCE, grid_res=0.1, bounds=((-1.75, 1.75), (-1.75, 1.75))):
    """
    Build an occupancy grid based on the positions of objects.
    Returns a 2D numpy array where 1 indicates an occupied cell and 0 indicates a free cell.
    Args:
        positions: Dictionary of object positions
        clearance: Clearance radius (meters)
        grid_res: Grid resolution (meters)
        bounds: ((x_min, x_max), (y_min, y_max))
    Returns:
        np.ndarray: Occupancy grid
    """
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # Calculate grid dimensions (10cm resolution)
    grid_width = int((x_max - x_min) / grid_res) + 1  # 36 cells (3.6m / 0.1m)
    grid_height = int((y_max - y_min) / grid_res) + 1  # 36 cells
    grid = np.zeros((grid_height, grid_width), dtype=int)
    
    # First mark the actual occupied cells
    for i in range(grid_height):
        for j in range(grid_width):
            world_x = j * grid_res + x_min
            world_y = i * grid_res + y_min
            if not is_point_safe([world_x, world_y], positions, 0):  # No clearance for initial marking
                grid[i, j] = 1
    
    # Then add clearance by marking surrounding cells
    clearance_cells = int(ROBOT_POSITION_CLEARANCE / grid_res)  # 0.25m / 0.1m = 2.5 -> 3 cells
    grid_with_clearance = grid.copy()
    
    # For each occupied cell, mark surrounding cells within clearance
    for i in range(grid_height):
        for j in range(grid_width):
            if grid[i, j] == 1:
                # Mark surrounding cells within clearance
                for di in range(-clearance_cells, clearance_cells + 1):
                    for dj in range(-clearance_cells, clearance_cells + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_height and 0 <= nj < grid_width:
                            if np.sqrt(di*di + dj*dj) <= clearance_cells:  # Circular clearance
                                grid_with_clearance[ni, nj] = 1
    
    return grid_with_clearance

def generate_waypoints(prev_last_waypoint, new_first_waypoint, positions):
    """
    Placeholder function for generating waypoints. Currently returns an empty list.
    Args:
        prev_last_waypoint: Last waypoint from previous scenario
        new_first_waypoint: First waypoint for new scenario
        positions: Object positions
    Returns:
        list: Empty list (to be implemented)
    """
    return []

def plot_scenario(ax, positions, waypoints, wall_rotation_deg=0, door_height=0.5, door_frame_info=None, wall_names=None, robot_position=None):
    """
    Plot a scenario with all objects, walls, door frame, waypoints, and robot position on a matplotlib axis.
    Args:
        ax: Matplotlib axis
        positions: Dictionary of object positions
        waypoints: List of waypoints
        wall_rotation_deg: Rotation of the walls in degrees
        door_height: Height of the door frame
        door_frame_info: Optional, door frame data
        wall_names: Optional, custom wall names
        robot_position: Optional, robot position to plot
    """
    wall_color = 'dimgray'
    wall_alpha = 0.5
    wall_rotation_rad = np.deg2rad(wall_rotation_deg)
    left_wall_total = 4.0
    left_wall_thickness = 0.1
    left_wall_segment_height = (left_wall_total - door_height) / 2
    wall_defs = [
        [0, 2.05, 4.2, 0.1, 0],           # Top wall
        [0, -2.05, 4.2, 0.1, 0],          # Bottom wall
        [2.05, 0, 0.1, 4, 0],             # Right wall
        [-2.05, (door_height/2) + (left_wall_segment_height/2), left_wall_thickness, left_wall_segment_height, 0],     # Left wall (upper)
        [-2.05, -(door_height/2) - (left_wall_segment_height/2), left_wall_thickness, left_wall_segment_height, 0],    # Left wall (lower)
    ]
    # Use default wall names if not provided
    if wall_names is None:
        wall_names = ['up_wall', 'down_wall', 'right_wall', 'left_wall_up', 'left_wall_bottom']
    for idx, (cx, cy, w, h, ang) in enumerate(wall_defs):
        rcx, rcy = rotate_point(cx, cy, wall_rotation_rad)
        wall_name = wall_names[idx]
        add_rotated_rect(ax, [rcx, rcy], w, h, ang + wall_rotation_rad, wall_color, wall_name, clearance=False, show_center=True, show_label=False)
    # Door frame
    door_center = [-2.05, 0.0]
    door_width = left_wall_thickness
    door_center_rot = rotate_point(door_center[0], door_center[1], wall_rotation_rad)
    add_rotated_rect(ax, door_center_rot, door_width, door_height, wall_rotation_rad, 'brown', 'Door Frame', clearance=False, show_center=True, show_label=True, rotation_angle_deg=np.degrees(wall_rotation_rad))
    # Movable objects
    for obj_name in ['box', 'chair', 'desk']:
        p = positions[obj_name]
        w, h = p['size']
        add_rotated_rect(ax, p['position'], w + MIN_CLEARANCE, h + MIN_CLEARANCE, p['rotation'], 'gray', None, clearance=True)
    add_rotated_rect(ax, positions['box']['position'], *positions['box']['size'], positions['box']['rotation'], 'blue', 'Box', show_center=True, show_label=True, rotation_angle_deg=np.degrees(positions['box']['rotation']))
    add_rotated_rect(ax, positions['chair']['position'], *positions['chair']['size'], positions['chair']['rotation'], 'green', 'Chair', show_center=True, show_label=True, rotation_angle_deg=np.degrees(positions['chair']['rotation']))
    add_rotated_rect(ax, positions['desk']['position'], *positions['desk']['size'], positions['desk']['rotation'], 'purple', 'Desk', show_center=True, show_label=True, rotation_angle_deg=np.degrees(positions['desk']['rotation']))
    # Plot waypoints (orange)
    if waypoints and len(waypoints) > 0:
        xs = [wp[0] for wp in waypoints]
        ys = [wp[1] for wp in waypoints]
        ax.plot(xs, ys, 'o-', color='orange', markersize=3, linewidth=1, label='Waypoints', zorder=6)
    # Plot robot position if provided
    if robot_position is not None:
        # Plot robot position as a red circle with clearance radius
        circle = plt.Circle(robot_position[:2], ROBOT_POSITION_CLEARANCE, color='red', alpha=0.3, label='Robot Clearance')
        ax.add_patch(circle)
        ax.plot(robot_position[0], robot_position[1], 'ro', markersize=5, label='Robot Position')
        # Draw a line indicating rotation (smaller and black)
        dx = 0.15 * np.cos(robot_position[2])  # Reduced from 0.2 to 0.15
        dy = 0.15 * np.sin(robot_position[2])
        ax.plot([robot_position[0], robot_position[0] + dx], 
                [robot_position[1], robot_position[1] + dy], 
                'k-', linewidth=1.5)  # Changed to black ('k') and slightly thicker
    # Highlight overlaps
    check_and_highlight_overlaps(ax, positions)
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-2.25, 2.25)
    ax.set_aspect('equal')
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc='center left', bbox_to_anchor=(1.01, 0.5),
        fontsize=6, frameon=False, borderaxespad=0
    ).set_visible(False)
    # Hide axis values
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

def get_corners_world(center, w, h, theta):
    """
    Calculate the world coordinates of the corners of a rectangle given its center, size, and rotation.
    Args:
        center (array): Center of rectangle
        w, h (float): Width and height
        theta (float): Rotation angle in radians
    Returns:
        list: List of corner coordinates in world frame
    """
    dx = w / 2
    dy = h / 2
    corners = np.array([
        [dx, dy],
        [-dx, dy],
        [-dx, -dy],
        [dx, -dy]
    ])
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    world_corners = np.dot(corners, rot.T) + center
    return world_corners.tolist()

def load_initial_scenario(ax0):
    """
    Load the initial scenario (scenario 0) from KNOWN_OBJECTS.
    Returns initial_positions, initial_waypoints, wall_export, scenario_data_0, and prev_waypoints.
    Args:
        ax0: Optional, matplotlib axis (not used)
    Returns:
        tuple: (initial_positions, initial_waypoints, wall_export, scenario_data_0, prev_waypoints)
    """
    initial_positions = {}
    for obj in KNOWN_OBJECTS:
        if obj.name in ["box", "chair", "desk"]:
            initial_positions[obj.name] = {
                "position": [obj.x, obj.y],
                "rotation": obj.rotation,
                "size": [obj.width, obj.depth]
            }
    initial_waypoints = []  # No waypoints
    left_wall_total = 4.0
    left_wall_thickness = 0.1
    left_wall_segment_height = (left_wall_total - 0.5) / 2
    wall_defs = [
        [0, 2.05, 4.2, 0.1, 0],           # Top wall
        [0, -2.05, 4.2, 0.1, 0],          # Bottom wall
        [2.05, 0, 0.1, 4, 0],             # Right wall
        [-2.05, (0.5/2) + (left_wall_segment_height/2), left_wall_thickness, left_wall_segment_height, 0],     # Left wall (upper)
        [-2.05, -(0.5/2) - (left_wall_segment_height/2), left_wall_thickness, left_wall_segment_height, 0],    # Left wall (lower)
    ]
    wall_names = ['up_wall', 'down_wall', 'right_wall', 'left_wall_up', 'left_wall_bottom']
    wall_export = []
    for idx, (cx, cy, w, h, ang) in enumerate(wall_defs):
        rcx, rcy = rotate_point(cx, cy, 0)
        wall_name = wall_names[idx]
        wall_export.append({
            "name": wall_name,
            "center": [rcx, rcy],
            "width": w,
            "height": h,
            "rotation": ang
        })
    door_center = [-2.05, 0.0]
    door_center_rot = rotate_point(door_center[0], door_center[1], 0)
    door_corners = get_corners_world(np.array(door_center_rot), left_wall_thickness, 0.5, 0)
    # Create ordered dictionary for objects
    ordered_objects = OrderedDict()
    for obj_name in ['box', 'chair', 'desk']:  # Maintain consistent order
        if obj_name in initial_positions:
            ordered_objects[obj_name] = {
                "center": initial_positions[obj_name]["position"],
                "position": initial_positions[obj_name]["position"],
                "rotation": initial_positions[obj_name]["rotation"],
                "width": initial_positions[obj_name]["size"][0],
                "height": initial_positions[obj_name]["size"][1],
                "corners": get_corners_world(
                    np.array(initial_positions[obj_name]["position"]),
                    initial_positions[obj_name]["size"][0],
                    initial_positions[obj_name]["size"][1],
                    initial_positions[obj_name]["rotation"]
                )
            }
    scenario_data_0 = {
        "scenario_id": 0,
        "objects": ordered_objects,
        "door_frame": {
            "center": door_center_rot,
            "position": door_center_rot,
            "rotation": 0,
            "width": left_wall_thickness,
            "height": 0.5,
            "corners": door_corners
        },
        "walls": wall_export,
        "waypoints": initial_waypoints
    }
    prev_waypoints = initial_waypoints
    return initial_positions, initial_waypoints, wall_export, scenario_data_0, prev_waypoints

scenario_id_counter = 0

def visualize_robot_positions(scenario_data, positions, ax, selected_positions=None, scenarios=None):
    """
    Visualize all robot positions for a base scenario, including scenario objects and all possible robot positions.
    Optionally highlights selected positions and their waypoints.
    Args:
        scenario_data: The base scenario data
        positions: Dictionary of object positions
        ax: Matplotlib axis to plot on
        selected_positions: List of selected robot positions (optional)
        scenarios: List of scenarios with waypoints (optional)
    """
    # Plot scenario objects
    for obj_name in ['box', 'chair', 'desk']:
        p = positions[obj_name]
        w, h = p['size']
        add_rotated_rect(ax, p['position'], w + MIN_CLEARANCE, h + MIN_CLEARANCE, p['rotation'], 'gray', None, clearance=True)
    add_rotated_rect(ax, positions['box']['position'], *positions['box']['size'], positions['box']['rotation'], 'blue', 'Box', show_center=True, show_label=True, rotation_angle_deg=np.degrees(positions['box']['rotation']))
    add_rotated_rect(ax, positions['chair']['position'], *positions['chair']['size'], positions['chair']['rotation'], 'green', 'Chair', show_center=True, show_label=True, rotation_angle_deg=np.degrees(positions['chair']['rotation']))
    add_rotated_rect(ax, positions['desk']['position'], *positions['desk']['size'], positions['desk']['rotation'], 'purple', 'Desk', show_center=True, show_label=True, rotation_angle_deg=np.degrees(positions['desk']['rotation']))
    
    # Plot walls
    wall_color = 'dimgray'
    wall_alpha = 0.5
    for wall in scenario_data["walls"]:
        add_rotated_rect(ax, wall["center"], wall["width"], wall["height"], wall["rotation"], wall_color, wall["name"], clearance=False, show_center=True, show_label=False)
    
    # Plot door frame
    door_frame = scenario_data["door_frame"]
    add_rotated_rect(ax, door_frame["center"], door_frame["width"], door_frame["height"], door_frame["rotation"], 'brown', 'Door Frame', clearance=False, show_center=True, show_label=True)
    
    # Plot all valid robot positions
    valid_positions = []
    grid = build_occupancy_grid(positions, clearance=ROBOT_POSITION_CLEARANCE, grid_res=0.1)
    grid_res = 0.1
    bounds = ((-1.75, 1.75), (-1.75, 1.75))
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 0:  # If cell is free
                world_x = j * grid_res + x_min
                world_y = i * grid_res + y_min
                valid_positions.append([world_x, world_y])
    
    # Plot valid positions as small dots
    if valid_positions:
        valid_x = [pos[0] for pos in valid_positions]
        valid_y = [pos[1] for pos in valid_positions]
        ax.scatter(valid_x, valid_y, c='red', s=10, alpha=0.3)
    
    # Plot selected positions if provided
    if selected_positions and scenarios:
        selected_x = [pos[0] for pos in selected_positions]
        selected_y = [pos[1] for pos in selected_positions]
        selected_theta = [pos[2] for pos in selected_positions]
        
        # Plot all robot positions in blue
        ax.scatter(selected_x, selected_y, c='blue', s=50, alpha=0.7)
        
        # Add numbers and rotation indicators to selected positions
        for i, (x, y, theta) in enumerate(zip(selected_x, selected_y, selected_theta)):
            ax.text(x, y, str(i+1), color='white', fontsize=8, ha='center', va='center')
            # Draw a line indicating rotation (smaller and black)
            dx = 0.15 * np.cos(theta)  # Reduced from 0.2 to 0.15
            dy = 0.15 * np.sin(theta)
            ax.plot([x, x + dx], [y, y + dy], 'k-', linewidth=1.5)  # Changed to black ('k') and slightly thicker
            
            # Plot waypoints for this position from the corresponding scenario
            if i < len(scenarios):
                waypoints = scenarios[i].get("waypoints", [])
                if waypoints:
                    # Determine color based on position (first or last)
                    if i == 0:  # First position
                        color = '#00FF00'  # Bright Green
                        label = 'First Waypoints' if i == 0 else ""
                    elif i == len(selected_positions) - 1:  # Last position
                        color = '#FF00FF'  # Magenta
                        label = 'Last Waypoints' if i == len(selected_positions) - 1 else ""
                    else:  # Middle positions
                        color = '#00CED1'  # Cyan
                        label = ""
                    
                    # First connect robot position to first waypoint
                    ax.plot([x, waypoints[0][0]], [y, waypoints[0][1]], '-', color=color, linewidth=1.5, label=label, zorder=6)
                    
                    # Plot waypoints with connecting lines and gradient colors
                    wp_xs = [wp[0] for wp in waypoints]
                    wp_ys = [wp[1] for wp in waypoints]
                    
                    # Plot connecting lines between waypoints
                    ax.plot(wp_xs, wp_ys, '-', color=color, linewidth=1.5, alpha=0.5, zorder=6)
                    
                    # Plot waypoint dots with decreasing opacity
                    n_waypoints = len(waypoints)
                    for j, (wx, wy) in enumerate(zip(wp_xs, wp_ys)):
                        # Calculate opacity based on position (first, last, or middle)
                        if i == 0:  # First position - increasing opacity
                            alpha = 0.5 + (0.5 * j / (n_waypoints - 1)) if n_waypoints > 1 else 1.0
                        elif i == len(selected_positions) - 1:  # Last position - decreasing opacity
                            alpha = 1.0 - (0.5 * j / (n_waypoints - 1)) if n_waypoints > 1 else 1.0
                        else:  # Middle positions - constant opacity
                            alpha = 0.7
                        ax.plot(wx, wy, 'o', color=color, markersize=5, alpha=alpha, zorder=6)
    
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-2.25, 2.25)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f"Scenario {scenario_data['scenario_id']}", fontsize=8)
    # Hide axis values
    ax.set_xticks([])
    ax.set_yticks([])

def get_discrete_heading():
    """
    Get one of the 8 discrete heading angles (N, NW, W, SW, S, SE, E, NE) in radians.
    Returns:
        float: Heading angle in radians
    """
    # Define the 8 discrete angles in radians
    angles = [
        0,          # N
        np.pi/4,    # NE
        np.pi/2,    # E
        3*np.pi/4,  # SE
        np.pi,      # S
        5*np.pi/4,  # SW
        3*np.pi/2,  # W
        7*np.pi/4   # NW
    ]
    return random.choice(angles)

def get_directions_for_heading(theta):
    """
    Get fixed directions based on discrete heading angles.
    Args:
        theta (float): Heading angle in radians
    Returns:
        list: List of (dx, dy) direction tuples
    """
    # Round theta to nearest discrete angle
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    theta = min(angles, key=lambda x: abs(x - theta))
    
    # Define directions for each discrete angle
    if theta == 0:  # N
        return [(0, 1), (0, 1), (-1, 1), (1, 1)]  # Forward (2x), Forward-left, Forward-right
    elif theta == np.pi/4:  # NE
        return [(1, 1), (1, 1), (0, 1), (1, 0)]  # Forward (2x), Forward-left, Forward-right
    elif theta == np.pi/2:  # E
        return [(1, 0), (1, 0), (1, -1), (1, 1)]  # Right (2x), Right-up, Right-down
    elif theta == 3*np.pi/4:  # SE
        return [(1, -1), (1, -1), (1, 0), (0, -1)]  # Forward (2x), Forward-left, Forward-right
    elif theta == np.pi:  # S
        return [(0, -1), (0, -1), (-1, -1), (1, -1)]  # Backward (2x), Backward-left, Backward-right
    elif theta == 5*np.pi/4:  # SW
        return [(-1, -1), (-1, -1), (0, -1), (-1, 0)]  # Forward (2x), Forward-left, Forward-right
    elif theta == 3*np.pi/2:  # W
        return [(-1, 0), (-1, 0), (-1, -1), (-1, 1)]  # Left (2x), Left-up, Left-down
    else:  # NW
        return [(-1, 1), (-1, 1), (-1, 0), (0, 1)]  # Forward (2x), Forward-left, Forward-right

def generate_robot_positions_for_scenario(scenario_data, prev_waypoints=None):
    """
    Generate multiple valid robot positions for a given scenario.
    Uses the occupancy grid to find all valid positions, then selects positions ensuring minimum distance constraints.
    For each selected position, generates waypoints starting from its neighbors.
    Returns a list of scenarios, each with a different valid robot position and its waypoints.
    Args:
        scenario_data: Scenario data dictionary
        prev_waypoints: Previous waypoints to avoid (optional)
    Returns:
        tuple: (duplicated_scenarios, attempts, distance_reductions, min_distance, positions_with_reduced_distance)
    """
    global scenario_id_counter
    duplicated_scenarios = []
    positions_with_reduced_distance = 0  # Track positions using reduced distance
    
    # Get the main scenario ID (0-9)
    main_scenario_id = scenario_data["scenario_id"]
    
    # Convert scenario objects to the format expected by is_point_safe
    positions = {
        obj_name: {
            "position": obj_data["position"],
            "rotation": obj_data["rotation"],
            "size": [obj_data["width"], obj_data["height"]]
        }
        for obj_name, obj_data in scenario_data["objects"].items()
    }
    
    # Add walls and door frame to positions for collision checking
    for wall in scenario_data["walls"]:
        positions[f"wall_{wall['name']}"] = {
            "position": wall["center"],
            "rotation": wall["rotation"],
            "size": [wall["width"], wall["height"]]
        }
    
    door_frame = scenario_data["door_frame"]
    positions["door_frame"] = {
        "position": door_frame["center"],
        "rotation": door_frame["rotation"],
        "size": [door_frame["width"], door_frame["height"]]
    }
    
    # Build occupancy grid once for all robot positions (10cm resolution)
    grid = build_occupancy_grid(positions, clearance=ROBOT_POSITION_CLEARANCE, grid_res=0.1)
    grid_res = 0.1  # 10cm resolution
    bounds = ((-1.75, 1.75), (-1.75, 1.75))
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # Find all valid positions at once
    valid_positions = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 0:  # If cell is free
                # Convert grid position to world coordinates
                world_x = j * grid_res + x_min
                world_y = i * grid_res + y_min
                # Check if position is far enough from previous waypoints
                if prev_waypoints and len(prev_waypoints) > 0:
                    last_wp = np.array(prev_waypoints[-1])
                    if np.linalg.norm(np.array([world_x, world_y]) - last_wp) < WAYPOINT_EXCLUSION_RADIUS:
                        continue
                valid_positions.append([world_x, world_y])
    
    # If we don't have enough valid positions, print warning and try with reduced clearance
    if len(valid_positions) < NUM_ROBOT_POSITION:
        # Try with reduced clearance
        reduced_clearance = ROBOT_POSITION_CLEARANCE * 0.5
        grid = build_occupancy_grid(positions, clearance=reduced_clearance, grid_res=0.1)
        valid_positions = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 0:
                    world_x = j * grid_res + x_min
                    world_y = i * grid_res + y_min
                    # Check if position is far enough from previous waypoints
                    if prev_waypoints and len(prev_waypoints) > 0:
                        last_wp = np.array(prev_waypoints[-1])
                        if np.linalg.norm(np.array([world_x, world_y]) - last_wp) < WAYPOINT_EXCLUSION_RADIUS:
                            continue
                    valid_positions.append([world_x, world_y])
        
        if len(valid_positions) < NUM_ROBOT_POSITION:
            return duplicated_scenarios, 0, 0, ROBOT_POSITION_CLEARANCE * 5
    
    # Select positions one by one, ensuring minimum distance between them
    selected_positions = []
    min_distance = ROBOT_POSITION_CLEARANCE * 5
    max_attempts = 1000  # Prevent infinite loops
    attempts = 0
    failed_positions = []  # Keep track of positions that failed distance check
    distance_reductions = 0  # Track number of distance reductions
    original_min_distance = min_distance  # Store original minimum distance
    
    # For initial scenario (scenario_id = 0), set first position to (0,0,0)
    if scenario_data["scenario_id"] == 0:
        # Check if (0,0) is a valid position
        is_valid = True
        for obj_name, obj_data in positions.items():
            if not is_point_safe([0, 0], {obj_name: obj_data}, ROBOT_POSITION_CLEARANCE):
                is_valid = False
                break
        
        if is_valid:
            # Add (0,0,0) as first position
            robot_position = [0, 0, 0]  # x=0, y=0, theta=0
            waypoints = generate_waypoints_for_position(robot_position, grid, grid_res, bounds)
            
            # Create scenario copy with selected position and waypoints
            duplicated_scenario = scenario_data.copy()
            duplicated_scenario["scenario_id"] = scenario_id_counter
            duplicated_scenario["main_scenario_id"] = main_scenario_id  # Add main scenario ID
            scenario_id_counter += 1
            duplicated_scenario["robot_position"] = robot_position
            duplicated_scenario["waypoints"] = waypoints
            duplicated_scenarios.append(duplicated_scenario)
            
            selected_positions.append(robot_position)
            # Remove (0,0) from valid positions if it exists
            valid_positions = [pos for pos in valid_positions if not (abs(pos[0]) < 1e-6 and abs(pos[1]) < 1e-6)]
    
    while len(selected_positions) < NUM_ROBOT_POSITION and len(valid_positions) > 0 and attempts < max_attempts:
        attempts += 1
        
        # Select a random position
        idx = random.randint(0, len(valid_positions) - 1)
        pos = valid_positions.pop(idx)
        
        # Check if this position is far enough from all already selected positions
        is_far_enough = True
        min_dist_found = float('inf')
        for selected_pos in selected_positions:
            # Only compare x,y coordinates (first two elements)
            dist = np.linalg.norm(np.array(pos) - np.array(selected_pos[:2]))
            min_dist_found = min(min_dist_found, dist)
            if dist < min_distance:
                is_far_enough = False
                break
        
        if is_far_enough:
            # Add discrete heading
            robot_position = [pos[0], pos[1], get_discrete_heading()]
            
            # Generate waypoints for this position
            waypoints = generate_waypoints_for_position(robot_position, grid, grid_res, bounds)
            
            # Create scenario copy with selected position and waypoints
            duplicated_scenario = scenario_data.copy()
            duplicated_scenario["scenario_id"] = scenario_id_counter
            duplicated_scenario["main_scenario_id"] = main_scenario_id  # Add main scenario ID
            scenario_id_counter += 1
            duplicated_scenario["robot_position"] = robot_position
            duplicated_scenario["waypoints"] = waypoints
            duplicated_scenarios.append(duplicated_scenario)
            
            selected_positions.append(robot_position)
            # Track if this position was accepted with reduced distance
            if min_distance < original_min_distance:
                positions_with_reduced_distance += 1
        else:
            failed_positions.append(pos)
            # If we've tried too many positions, try with reduced distance requirement
            if len(failed_positions) > len(valid_positions) * 0.5:
                if min_distance > original_min_distance * 0.8:  # Only reduce if not already at 80%
                    min_distance *= 0.8  # Reduce minimum distance by 20%
                    distance_reductions += 1
                    # Put failed positions back in the pool
                    valid_positions.extend(failed_positions)
                    failed_positions = []
    
    return duplicated_scenarios, attempts, distance_reductions, min_distance, positions_with_reduced_distance

def generate_scenario(ax, scenario_count, prev_waypoints):
    """
    Generate a single scenario, including wall definitions, door frame, scenario positions, and plotting.
    Returns the scenario data and the updated prev_waypoints.
    Args:
        ax: Optional, matplotlib axis (not used)
        scenario_count: Index of the scenario
        prev_waypoints: Previous waypoints to avoid
    Returns:
        tuple: (scenario_data, prev_waypoints)
    """
    tries = 0
    wall_rotation_deg = random.choice([0, 90, 180, 270])
    door_height = random.uniform(0.5, 1)
    left_wall_thickness = 0.1
    left_wall_total = 4.0
    left_wall_segment_height = (left_wall_total - door_height) / 2
    wall_rotation_rad = np.deg2rad(wall_rotation_deg)
    # Wall definitions: (center_x, center_y, width, height, angle_rad)
    wall_defs = [
        [0, 2.05, 4.2, 0.1, 0],           # Top wall
        [0, -2.05, 4.2, 0.1, 0],          # Bottom wall
        [2.05, 0, 0.1, 4, 0],             # Right wall
        [-2.05, (door_height/2) + (left_wall_segment_height/2), left_wall_thickness, left_wall_segment_height, 0],     # Left wall (upper)
        [-2.05, -(door_height/2) - (left_wall_segment_height/2), left_wall_thickness, left_wall_segment_height, 0],    # Left wall (lower)
    ]
    wall_names = ['up_wall', 'down_wall', 'right_wall', 'left_wall_up', 'left_wall_bottom']
    wall_export = []
    for idx, (cx, cy, w, h, ang) in enumerate(wall_defs):
        rcx, rcy = rotate_point(cx, cy, wall_rotation_rad)
        wall_name = wall_names[idx]
        wall_export.append({
            "name": wall_name,
            "center": [rcx, rcy],
            "width": w,
            "height": h,
            "rotation": ang + wall_rotation_rad
        })
    door_center = [-2.05, 0.0]
    door_center_rot = rotate_point(door_center[0], door_center[1], wall_rotation_rad)
    door_corners = get_corners_world(np.array(door_center_rot), left_wall_thickness, door_height, wall_rotation_rad)
    while True:
        positions = generate_scenario_positions(max_order_tries=10, scenario_index=scenario_count, prev_waypoints=prev_waypoints)
        tries += 1
        if positions is not None:
            vbox_waypoints = []  # No v_box waypoints
            connecting_waypoints = generate_waypoints(prev_waypoints[-1] if prev_waypoints else None, vbox_waypoints[0] if vbox_waypoints else None, positions)
            # Retry scenario generation if no A* path is found, up to 10 times
            if prev_waypoints and vbox_waypoints and len(connecting_waypoints) == 0:
                print(f"[DEBUG] No A* path found for scenario {scenario_count} from ({prev_waypoints[-1][0]:.2f}, {prev_waypoints[-1][1]:.2f}) to ({vbox_waypoints[0][0]:.2f}, {vbox_waypoints[0][1]:.2f}), retrying ({tries}/10)...")
                if tries < 10:
                    continue
            # Concatenate connecting and vbox waypoints for full path
            full_waypoints = connecting_waypoints + vbox_waypoints
            # Create ordered dictionary for objects
            ordered_objects = OrderedDict()
            for obj_name in ['box', 'chair', 'desk']:  # Maintain consistent order
                if obj_name in positions:
                    ordered_objects[obj_name] = {
                        "center": positions[obj_name]["position"],
                        "position": positions[obj_name]["position"],
                        "rotation": positions[obj_name]["rotation"],
                        "width": positions[obj_name]["size"][0],
                        "height": positions[obj_name]["size"][1],
                        "corners": get_corners_world(
                            np.array(positions[obj_name]["position"]),
                            positions[obj_name]["size"][0],
                            positions[obj_name]["size"][1],
                            positions[obj_name]["rotation"]
                        )
                    }
            scenario_data = {
                "scenario_id": scenario_count,
                "objects": ordered_objects,
                "door_frame": {
                    "center": door_center_rot,
                    "position": door_center_rot,
                    "rotation": wall_rotation_rad,
                    "width": left_wall_thickness,
                    "height": door_height,
                    "corners": door_corners
                },
                "walls": wall_export,
                "waypoints": full_waypoints
            }
            prev_waypoints = vbox_waypoints
            return scenario_data, prev_waypoints
        elif tries > 20:
            return None, prev_waypoints

def round_floats(obj):
    """
    Recursively round all floats in a nested structure (dict, list, float, numpy float).
    Args:
        obj: Any object (float, list, dict, etc.)
    Returns:
        Rounded object
    """
    if isinstance(obj, float):
        return round(obj, 2)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return round(float(obj), 2)
    elif isinstance(obj, list):
        return [round_floats(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    else:
        return obj

def try_direction(theta, current_x, current_y, grid, cone_angle=0.2):
    """
    Try to find a valid waypoint in the given direction or nearby cells.
    Args:
        theta: Target direction angle
        current_x: Current x position in grid coordinates
        current_y: Current y position in grid coordinates
        grid: The occupancy grid
        cone_angle: Half-angle of the cone to search (in radians)
    Returns:
        tuple: (nx, ny, success) where nx, ny are grid coordinates and success is a bool
    """
    # First try exact direction
    dx = np.cos(theta)
    dy = np.sin(theta)
    grid_dx = int(round(dy))
    grid_dy = int(round(dx))
    nx, ny = current_x + grid_dx, current_y + grid_dy
    
    if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 
        grid[nx, ny] == 0):
        return nx, ny, True
    
    # If exact direction fails, try nearby cells in a cone
    # Try 3 angles on each side of the target direction
    for angle_offset in [-cone_angle, -cone_angle/2, -cone_angle/4, 
                        cone_angle/4, cone_angle/2, cone_angle]:
        test_theta = theta + angle_offset
        dx = np.cos(test_theta)
        dy = np.sin(test_theta)
        grid_dx = int(round(dy))
        grid_dy = int(round(dx))
        nx, ny = current_x + grid_dx, current_y + grid_dy
        
        if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 
            grid[nx, ny] == 0):
            return nx, ny, True
    
    return None, None, False

def generate_waypoints_for_position(robot_position, grid, grid_res=0.1, bounds=((-1.75, 1.75), (-1.75, 1.75))):
    """
    Generate N waypoints for a given robot position using the occupancy grid.
    If WAYPOINTS_RELATIVE_TO_HEADING is True:
    - First waypoint will be exactly in the direction of robot heading
    - If heading direction is blocked, tries opposite direction
    - Subsequent waypoints will be in fixed directions based on discrete heading angles
    Args:
        robot_position: [x, y, theta] of the robot
        grid: The occupancy grid (0 for free, 1 for occupied)
        grid_res: Grid resolution in meters
        bounds: World coordinate bounds
    Returns:
        List of N waypoints in world coordinates
    """
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # Convert robot position to grid coordinates
    robot_x, robot_y = robot_position[:2]
    robot_theta = robot_position[2]
    
    # Convert to grid coordinates (note: grid y is world x and grid x is world y)
    robot_grid_x = int((robot_y - y_min) / grid_res)
    robot_grid_y = int((robot_x - x_min) / grid_res)
    
    waypoints = []
    current_x, current_y = robot_grid_x, robot_grid_y
    current_theta = robot_theta
    
    for i in range(NUM_WAYPOINTS_PER_POSITION):
        if WAYPOINTS_RELATIVE_TO_HEADING:
            if i == 0:  # First waypoint
                # First waypoint: try robot heading direction first with cone search
                nx, ny, success = try_direction(current_theta, current_x, current_y, grid, cone_angle=0.2)
                
                if not success:
                    # If heading direction is blocked, try opposite direction with cone search
                    opposite_theta = (current_theta + np.pi) % (2 * np.pi)
                    nx, ny, success = try_direction(opposite_theta, current_x, current_y, grid, cone_angle=0.2)
                    
                    if success:
                        # Update current_theta to opposite direction for subsequent waypoints
                        current_theta = opposite_theta
                
                if success:
                    current_x, current_y = nx, ny
                else:
                    # If both directions are blocked, try all 8 directions with cone search
                    for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
                        nx, ny, success = try_direction(angle, current_x, current_y, grid, cone_angle=0.2)
                        if success:
                            current_x, current_y = nx, ny
                            current_theta = angle
                            break
                    
                    if not success:
                        break  # No valid position found, stop generating waypoints
            else:
                # Get directions based on current heading
                directions = get_directions_for_heading(current_theta)
                
                # Find valid neighbors in these directions
                valid_neighbors = []
                for dx, dy in directions:
                    nx, ny = current_x + dx, current_y + dy
                    if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 
                        grid[nx, ny] == 0):
                        valid_neighbors.append((nx, ny))
                
                if not valid_neighbors:
                    break  # No valid neighbors found, stop generating waypoints
                
                # Randomly select one of the valid neighbors
                current_x, current_y = random.choice(valid_neighbors)
        else:
            # Original 8-connected grid behavior
            directions = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
            valid_neighbors = []
            for dx, dy in directions:
                nx, ny = current_x + dx, current_y + dy
                if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 
                    grid[nx, ny] == 0):
                    valid_neighbors.append((nx, ny))
            
            if not valid_neighbors:
                break  # No valid neighbors found, stop generating waypoints
            
            # Randomly select one of the valid neighbors
            current_x, current_y = random.choice(valid_neighbors)
        
        # Convert grid coordinates back to world coordinates
        world_x = current_y * grid_res + x_min
        world_y = current_x * grid_res + y_min
        
        # Calculate new heading based on movement direction
        if len(waypoints) > 0:
            prev_x, prev_y = waypoints[-1]
            dx = world_x - prev_x
            dy = world_y - prev_y
            current_theta = np.arctan2(dy, dx)
        
        waypoints.append([world_x, world_y])
    
    return waypoints

# --- Main Execution Block ---
if __name__ == "__main__":
    # Calculate grid dimensions for scenarios
    ncols = min(4, NUM_SCENARIOS)  # Maximum 4 columns
    nrows = (NUM_SCENARIOS + ncols - 1) // ncols  # Ceiling division
    
    # Create a figure for robot positions visualization with dynamic grid
    fig_positions, axes_positions = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    if NUM_SCENARIOS == 1:
        axes_positions = np.array([[axes_positions]])
    elif nrows == 1:
        axes_positions = axes_positions.reshape(1, -1)
    elif ncols == 1:
        axes_positions = axes_positions.reshape(-1, 1)
    
    # Use the new function to load the initial scenario
    initial_positions, initial_waypoints, wall_export, scenario_data_0, prev_waypoints = load_initial_scenario(None)
    
    # Get selected positions for initial scenario
    duplicated_scenarios, attempts, reductions, min_dist, reduced_positions = generate_robot_positions_for_scenario(scenario_data_0)
    selected_positions = [scenario["robot_position"] for scenario in duplicated_scenarios]
    
    # Visualize robot positions for initial scenario
    positions = {
        obj_name: {
            "position": obj_data["position"],
            "rotation": obj_data["rotation"],
            "size": [obj_data["width"], obj_data["height"]]
        }
        for obj_name, obj_data in scenario_data_0["objects"].items()
    }
    visualize_robot_positions(scenario_data_0, positions, axes_positions[0, 0], selected_positions, duplicated_scenarios)
    
    all_scenarios = []
    all_scenarios.extend(duplicated_scenarios)
    
    scenario_count = 1
    total_attempts = attempts
    total_reductions = reductions
    final_min_distance = min_dist
    total_reduced_positions = reduced_positions

    # Main scenario loop
    for idx in tqdm(range(1, NUM_SCENARIOS), desc="Generating scenarios"):
        scenario_data, prev_waypoints = generate_scenario(None, scenario_count, prev_waypoints)
        if scenario_data:
            # Get selected positions for this scenario
            duplicated_scenarios, attempts, reductions, min_dist, reduced_positions = generate_robot_positions_for_scenario(scenario_data)
            selected_positions = [scenario["robot_position"] for scenario in duplicated_scenarios]
            
            # Update totals
            total_attempts += attempts
            total_reductions += reductions
            final_min_distance = min(final_min_distance, min_dist)  # Keep track of the smallest distance used
            total_reduced_positions += reduced_positions
            
            # Visualize robot positions for this scenario
            positions = {
                obj_name: {
                    "position": obj_data["position"],
                    "rotation": obj_data["rotation"],
                    "size": [obj_data["width"], obj_data["height"]]
                }
                for obj_name, obj_data in scenario_data["objects"].items()
            }
            row = scenario_count // ncols
            col = scenario_count % ncols
            visualize_robot_positions(scenario_data, positions, axes_positions[row, col], selected_positions, duplicated_scenarios)
            
            all_scenarios.extend(duplicated_scenarios)
            scenario_count += 1
    
    # Hide unused axes if any
    for idx in range(scenario_count, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes_positions[row, col].axis('off')
    
    # Save the robot positions visualization first
    plt.figure(fig_positions.number)
    plt.tight_layout()
    os.makedirs("output/senarios/plots", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    positions_plot_filename = f"output/senarios/plots/{len(all_scenarios)}_robot_positions_{timestamp}.png"
    plt.savefig(positions_plot_filename, dpi=300)
    plt.show()
    
    # Create and show the main visualization only if enabled
    if SHOW_MAIN_VISUALIZATION:
        total_scenarios = NUM_SCENARIOS * NUM_ROBOT_POSITION
        ncols = int(np.ceil(np.sqrt(total_scenarios)))
        nrows = int(np.ceil(total_scenarios / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape((nrows, ncols))
        
        # Plot all scenarios including initial and its duplicates
        for i, scenario in enumerate(all_scenarios):
            ax = axes[i // ncols, i % ncols]
            positions = {
                obj_name: {
                    "position": obj_data["position"],
                    "rotation": obj_data["rotation"],
                    "size": [obj_data["width"], obj_data["height"]]
                }
                for obj_name, obj_data in scenario["objects"].items()
            }
            plot_scenario(ax, positions, scenario["waypoints"], 
                         wall_rotation_deg=0, door_height=0.5,
                         robot_position=scenario.get("robot_position"))
            ax.set_title(f"Scenario {scenario['scenario_id']}", fontsize=8)
        
        # Hide unused axes if any
        for idx in range(total_scenarios, nrows * ncols):
            i = idx // ncols
            j = idx % ncols
            axes[i, j].axis('off')
        
        # Save the main visualization
        plt.figure(fig.number)
        plt.tight_layout()
        plot_filename = f"output/senarios/plots/{len(all_scenarios)}_scenarios_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300)
        plt.show()
    
    # Export scenarios to JSONL
    os.makedirs("output/senarios", exist_ok=True)
    filename = f"output/senarios/{len(all_scenarios)}_scenarios_{timestamp}.jsonl"
    with open(filename, "w") as f:
        for scenario in all_scenarios:
            # Ensure robot_position is properly formatted with theta
            if "robot_position" in scenario:
                scenario["robot_position"] = [
                    round(scenario["robot_position"][0], 2),
                    round(scenario["robot_position"][1], 2),
                    round(scenario["robot_position"][2], 2)  # Ensure theta is included
                ]
            scenario_rounded = round_floats(scenario)
            f.write(json.dumps(scenario_rounded) + "\n")

    print(f"\nFinal Report:")
    print(f"Total scenarios generated: {len(all_scenarios)}")
    print(f"Number of robot positions per scenario: {NUM_ROBOT_POSITION}")
    print(f"Original minimum distance: {ROBOT_POSITION_CLEARANCE * 5:.2f} units")
    print(f"Final minimum distance used: {final_min_distance:.2f} units")
    print(f"Total attempts made: {total_attempts}")
    print(f"Number of distance reductions: {total_reductions}")
    print(f"Positions using reduced distance: {total_reduced_positions} ({(total_reduced_positions/len(all_scenarios)*100):.1f}%)")
    print(f"Output files:")
    print(f"- Scenarios: {filename}")
    print(f"- Robot positions visualization: {positions_plot_filename}")
    if SHOW_MAIN_VISUALIZATION:
        print(f"- Main visualization: {plot_filename}")