import math
import numpy as np

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

class LidarScan:
    """
    Utility class for handling LiDAR scan data and reconstructing angles.
    """
    def __init__(self, raw_scan, pose, frame_id=None, lidar_fov=2*np.pi):
        self.raw_scan = np.array(raw_scan)
        self.pose = pose
        self.frame_id = frame_id
        self.lidar_fov = lidar_fov
        self.lidar_width = len(self.raw_scan)
        self.angles = self._compute_angles()

    def _compute_angles(self):
        angle_increment = self.lidar_fov / self.lidar_width
        return np.array([(i * angle_increment + np.pi) % (2 * np.pi) for i in range(self.lidar_width)])

# Define known objects in the environment
KNOWN_OBJECTS = [
    Object2D(x=1, y=1, width=0.4, depth=0.4, name="chair"),
    Object2D(x=-1, y=1, width=0.4, depth=0.6, name="box"),
    Object2D(x=0.7, y=-0.9, width=0.8, depth=1.6, name="desk", rotation=math.pi/2),
    Object2D(x=-0, y=0, width=0.3, depth=0.3, name="v_box", should_scan=False),
    Object2D(x=-2.05, y=0, width=0.1, depth=0.5, name="door_frame", should_scan=False)
] 