import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms
from glob import glob

def add_rotated_rect(ax, center, width, height, angle_rad, color, label, show_center=True, show_label=True, rotation_angle_deg=None):
    rect = Rectangle((center[0] - width/2, center[1] - height/2), width, height, angle=0, color=color, alpha=0.5, label=label)
    t = mtransforms.Affine2D().rotate_around(center[0], center[1], angle_rad) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    if show_center:
        ax.plot(center[0], center[1], 'ko', markersize=3, zorder=10)
    if show_label:
        ax.text(center[0], center[1], f'{width:.2f}×{height:.2f}', color='black', fontsize=7, ha='center', va='center', zorder=11, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
    if rotation_angle_deg is not None:
        ax.text(center[0], center[1]-0.13, f'{rotation_angle_deg:.0f}°', color='black', fontsize=7, ha='center', va='center', zorder=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))

def plot_scenario_from_json(ax, scenario):
    # Walls
    for wall in scenario.get('walls', []):
        add_rotated_rect(
            ax,
            wall['center'],
            wall['width'],
            wall['height'],
            wall['rotation'],
            'dimgray',
            None,  # No label for wall
            show_center=False,
            show_label=False
        )
        # Optionally, to debug wall names visually, uncomment the next line:
        # ax.text(wall['center'][0], wall['center'][1], wall.get('name', ''), color='red', fontsize=9, ha='center', va='center', zorder=20, bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.1'))
    # Door frame
    df = scenario['door_frame']
    add_rotated_rect(ax, df['center'], df['width'], df['height'], df['rotation'], 'brown', 'Door Frame', show_center=True, show_label=True, rotation_angle_deg=np.degrees(df['rotation']))
    # Objects
    colors = {'box': 'blue', 'chair': 'green', 'desk': 'purple', 'v_box': 'red'}
    for obj_name in ['box', 'chair', 'desk', 'v_box']:
        obj = scenario['objects'][obj_name]
        rot_deg = np.degrees(obj['rotation']) if obj_name in ['box', 'chair', 'desk'] else None
        add_rotated_rect(ax, obj['center'], obj['width'], obj['height'], obj['rotation'], colors[obj_name], obj_name.capitalize(), show_center=True, show_label=True, rotation_angle_deg=rot_deg)
    # Waypoints
    waypoints = scenario.get('waypoints', [])
    if waypoints:
        xs = [wp[0] for wp in waypoints]
        ys = [wp[1] for wp in waypoints]
        ax.plot(xs, ys, 'o-', color='orange', markersize=3, linewidth=1, label='Waypoints', zorder=5)
        ax.plot(xs[0], ys[0], 'ro', markersize=7, label='Target Waypoint', zorder=6)
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-2.25, 2.25)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend().set_visible(False)

def find_latest_jsonl(directory):
    files = sorted(glob(os.path.join(directory, '*.jsonl')), key=os.path.getmtime, reverse=True)
    return files[0] if files else None

def main():
    jsonl_path = find_latest_jsonl('output/senarios')
    if not jsonl_path:
        print('No JSONL file found in output/senarios/')
        return
    with open(jsonl_path, 'r') as f:
        scenarios = [json.loads(line) for line in f]
    num_scenarios = len(scenarios)
    ncols = int(np.ceil(np.sqrt(num_scenarios)))
    nrows = int(np.ceil(num_scenarios / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape((nrows, ncols))
    for idx, scenario in enumerate(scenarios):
        i = idx // ncols
        j = idx % ncols
        ax = axes[i, j]
        plot_scenario_from_json(ax, scenario)
        ax.set_title(f"Scenario {scenario['scenario_id']}", fontsize=8)
    for idx in range(num_scenarios, nrows * ncols):
        i = idx // ncols
        j = idx % ncols
        axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 