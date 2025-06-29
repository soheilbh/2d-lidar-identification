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
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from bot3.utils.lidar_utils import KNOWN_OBJECTS

NUM_SCENARIOS = 36  # Number of scenarios to generate

# Object sizes
SIZES = {
    'box': (0.4, 0.6),
    'chair': (0.4, 0.4),
    'desk': (0.8, 1.6),
    'v_box': (0.3, 0.3)
}
MIN_CLEARANCE = 0.6

WAYPOINT_EXCLUSION_RADIUS = 0.6  # meters
ASTAR_CLEARANCE = 0.15  # meters, used for A* path planning

# Helper to get randomized size for an object
def get_randomized_size(obj_name):
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
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
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
            # v_box
            v_box_pos = generate_random_position(existing_positions, 'v_box', scenario_index, obj_size=SIZES['v_box'], prev_waypoints=prev_waypoints)
            if v_box_pos['position'] == [0.0, 0.0] and len(existing_positions) > 0:
                continue
            positions['v_box'] = {
                'position': v_box_pos['position'],
                'rotation': v_box_pos['rotation'],
                'size': v_box_pos['size']
            }
            return positions
    return None

def generate_waypoints_for_vbox(vbox):
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
            ax.text(center[0], center[1]-0.13, f'{rotation_angle_deg:.0f}°', color='black', fontsize=7, ha='center', va='center', zorder=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))

def check_and_highlight_overlaps(ax, positions):
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
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return [c * x - s * y, s * x + c * y]

def is_point_safe(point, positions, clearance):
    px, py = point
    leg_radius = 0.15  # meters
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
            # Full rectangle occupied (box, v_box)
            w_clear = obj_size[0] + clearance
            h_clear = obj_size[1] + clearance
            if rectangles_overlap(
                np.array([px, py]), clearance, clearance, 0,
                obj_pos, w_clear, h_clear, obj_data['rotation']
            ):
                return False
    return corners_within_bounds(np.array([px, py]), clearance, clearance, 0)

def astar_path(start, goal, positions, clearance=ASTAR_CLEARANCE, grid_res=0.2, bounds=((-1.8, 1.8), (-1.8, 1.8))):
    """
    A* pathfinding from start to goal, avoiding obstacles with given clearance.
    Returns a list of waypoints (including start and goal).
    """
    from collections import deque
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    def to_grid(p):
        return (int(round((p[0] - x_min) / grid_res)), int(round((p[1] - y_min) / grid_res)))
    def to_world(g):
        return (g[0] * grid_res + x_min, g[1] * grid_res + y_min)
    start_g = to_grid(start)
    goal_g = to_grid(goal)
    open_set = []
    heapq.heappush(open_set, (0, start_g))
    came_from = {}
    g_score = {start_g: 0}
    f_score = {start_g: np.linalg.norm(np.array(start) - np.array(goal))}
    visited = set()
    max_iter = 100000  # Increased from 50000
    for _ in range(max_iter):
        if not open_set:
            break
        _, current = heapq.heappop(open_set)
        if current == goal_g:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path = path[::-1]
            return [list(to_world(g)) for g in path]
        visited.add(current)
        cx, cy = to_world(current)
        # Add diagonal moves with smaller step size
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-0.5,-0.5),(-0.5,0.5),(0.5,-0.5),(0.5,0.5)]:
            neighbor = (current[0]+dx, current[1]+dy)
            nx, ny = to_world(neighbor)
            if not (x_min <= nx <= x_max and y_min <= ny <= y_max):
                continue
            if neighbor in visited:
                continue
            if not is_point_safe([nx, ny], positions, clearance):
                continue
            tentative_g = g_score[current] + np.linalg.norm(np.array([cx, cy]) - np.array([nx, ny]))
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + np.linalg.norm(np.array([nx, ny]) - np.array(goal))
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []  # No path found

def generate_connecting_waypoints(prev_last_waypoint, new_first_waypoint, positions):
    """
    Generate safe waypoints to connect the last waypoint of previous scenario to the first waypoint of new scenario.
    Uses A* path planning to ensure clearance from obstacles.
    Returns all intermediate waypoints at 20cm intervals (grid size), excluding the start but including the end.
    """
    if prev_last_waypoint is None or new_first_waypoint is None:
        return []
    
    # Calculate direct distance between waypoints
    direct_dist = np.linalg.norm(np.array(prev_last_waypoint) - np.array(new_first_waypoint))
    
    # If waypoints are too close, generate intermediate points
    if direct_dist < 0.3:  # If less than 30cm apart
        mid_point = [(prev_last_waypoint[0] + new_first_waypoint[0])/2,
                    (prev_last_waypoint[1] + new_first_waypoint[1])/2]
        # Try to find a safe intermediate point
        for angle in np.linspace(0, 2*np.pi, 8):
            offset = 0.2  # 20cm offset
            test_point = [
                mid_point[0] + offset * np.cos(angle),
                mid_point[1] + offset * np.sin(angle)
            ]
            if is_point_safe(test_point, positions, ASTAR_CLEARANCE):
                return [test_point]
        return []
    
    # For longer distances, use A* path planning
    path = astar_path(prev_last_waypoint, new_first_waypoint, positions, clearance=ASTAR_CLEARANCE)
    
    # If no path found, try with reduced clearance
    if not path:
        path = astar_path(prev_last_waypoint, new_first_waypoint, positions, clearance=ASTAR_CLEARANCE*0.8)
    
    # Return all intermediate waypoints (exclude start, include end)
    return path[1:] if len(path) > 1 else []

def plot_scenario(ax, positions, waypoints, wall_rotation_deg=0, door_height=0.5, door_frame_info=None, wall_names=None):
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
    for obj_name in ['box', 'chair', 'desk', 'v_box']:
        p = positions[obj_name]
        w, h = p['size']
        add_rotated_rect(ax, p['position'], w + MIN_CLEARANCE, h + MIN_CLEARANCE, p['rotation'], 'gray', None, clearance=True)
    add_rotated_rect(ax, positions['box']['position'], *positions['box']['size'], positions['box']['rotation'], 'blue', 'Box', show_center=True, show_label=True, rotation_angle_deg=np.degrees(positions['box']['rotation']))
    add_rotated_rect(ax, positions['chair']['position'], *positions['chair']['size'], positions['chair']['rotation'], 'green', 'Chair', show_center=True, show_label=True, rotation_angle_deg=np.degrees(positions['chair']['rotation']))
    add_rotated_rect(ax, positions['desk']['position'], *positions['desk']['size'], positions['desk']['rotation'], 'purple', 'Desk', show_center=True, show_label=True, rotation_angle_deg=np.degrees(positions['desk']['rotation']))
    add_rotated_rect(ax, positions['v_box']['position'], *positions['v_box']['size'], positions['v_box']['rotation'], 'red', 'Virtual Box', show_center=True, show_label=True)
    # Plot waypoints (blue)
    if waypoints and len(waypoints) > 0:
        xs = [wp[0] for wp in waypoints]
        ys = [wp[1] for wp in waypoints]
        ax.plot(xs, ys, 'o-', color='blue', markersize=3, linewidth=1, label='Path', zorder=6)
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

def get_corners_world(center, w, h, theta):
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

if __name__ == "__main__":
    # Compute grid size for visualization
    ncols = int(np.ceil(np.sqrt(NUM_SCENARIOS)))
    nrows = int(np.ceil(NUM_SCENARIOS / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape((nrows, ncols))
    all_scenarios = []
    prev_waypoints = None

    # Generate initial scenario 0 from KNOWN_OBJECTS
    ax0 = axes[0, 0]
    initial_positions = {}
    for obj in KNOWN_OBJECTS:
        if obj.name in ["box", "chair", "desk", "v_box"]:
            initial_positions[obj.name] = {
                "position": [obj.x, obj.y],
                "rotation": obj.rotation,
                "size": [obj.width, obj.depth]
            }
    initial_waypoints = generate_waypoints_for_vbox(initial_positions['v_box']) if 'v_box' in initial_positions else []
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
    plot_scenario(ax0, initial_positions, initial_waypoints, wall_rotation_deg=0, door_height=0.5)
    ax0.set_title("Scenario 0", fontsize=8)
    scenario_data_0 = {
        "scenario_id": 0,
        "objects": {
            obj: {
                "center": initial_positions[obj]["position"],
                "position": initial_positions[obj]["position"],
                "rotation": initial_positions[obj]["rotation"],
                "width": initial_positions[obj]["size"][0],
                "height": initial_positions[obj]["size"][1],
                "corners": get_corners_world(
                    np.array(initial_positions[obj]["position"]),
                    initial_positions[obj]["size"][0],
                    initial_positions[obj]["size"][1],
                    initial_positions[obj]["rotation"]
                )
            } for obj in initial_positions
        },
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
    all_scenarios.append(scenario_data_0)
    prev_waypoints = initial_waypoints
    scenario_count = 1

    # Main scenario loop (start from 1, generate NUM_SCENARIOS-1 more)
    for idx in tqdm(range(1, NUM_SCENARIOS), desc="Generating scenarios"):
        i = idx // ncols
        j = idx % ncols
        ax = axes[i, j]
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
                vbox_waypoints = generate_waypoints_for_vbox(positions['v_box'])
                connecting_waypoints = generate_connecting_waypoints(prev_waypoints[-1] if prev_waypoints else None, vbox_waypoints[0] if vbox_waypoints else None, positions)
                # Retry scenario generation if no A* path is found, up to 10 times
                if prev_waypoints and vbox_waypoints and len(connecting_waypoints) == 0:
                    print(f"[DEBUG] No A* path found for scenario {scenario_count} from {prev_waypoints[-1]} to {vbox_waypoints[0]}, retrying ({tries}/10)...")
                    if tries < 10:
                        continue
                # Concatenate connecting and vbox waypoints for full path
                full_waypoints = connecting_waypoints + vbox_waypoints
                plot_scenario(ax, positions, full_waypoints, wall_rotation_deg=wall_rotation_deg, door_height=door_height)
                ax.set_title(f"Scenario {scenario_count}", fontsize=8)
                scenario_data = {
                    "scenario_id": scenario_count,
                    "objects": {
                        obj: {
                            "center": positions[obj]["position"],
                            "position": positions[obj]["position"],
                            "rotation": positions[obj]["rotation"],
                            "width": positions[obj]["size"][0],
                            "height": positions[obj]["size"][1],
                            "corners": get_corners_world(
                                np.array(positions[obj]["position"]),
                                positions[obj]["size"][0],
                                positions[obj]["size"][1],
                                positions[obj]["rotation"]
                            )
                        } for obj in positions
                    },
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
                all_scenarios.append(scenario_data)
                scenario_count += 1
                prev_waypoints = vbox_waypoints
                break
            elif tries > 20:
                ax.set_title("SKIPPED", fontsize=8)
                ax.axis('off')
                break
    # Hide unused axes if any
    for idx in range(NUM_SCENARIOS, nrows * ncols):
        i = idx // ncols
        j = idx % ncols
        axes[i, j].axis('off')
    plt.tight_layout()
    # Export scenarios to JSONL
    os.makedirs("output/senarios", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/senarios/{len(all_scenarios)}_scenarios_{timestamp}.jsonl"
    with open(filename, "w") as f:
        for scenario in all_scenarios:
            scenario = json.loads(json.dumps(scenario, default=lambda o: float(o) if isinstance(o, (np.floating, np.float32, np.float64)) else o))
            f.write(json.dumps(scenario) + "\n")
    print(f"Exported {len(all_scenarios)} scenarios to {filename}")
    # Save the final plot as an image
    os.makedirs("output/senarios/plots", exist_ok=True)
    plot_filename = f"output/senarios/plots/{len(all_scenarios)}_scenarios_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved final plot to {plot_filename}")
    plt.show() 