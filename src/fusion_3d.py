"""
2D-to-3D Fusion Module
Projects 2D bounding boxes onto depth maps for 3D localization
Uses pinhole camera model
"""

import numpy as np
import cv2

class CameraIntrinsics:
    """
    Pinhole camera model parameters
    Typical values for RGB-D cameras (Kinect/RealSense)
    """
    def __init__(self, fx=525.0, fy=525.0, cx=112.0, cy=74.0):
        """
        Args:
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point (image center)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def pixel_to_3d(self, u, v, depth):
        """
        Convert pixel (u,v) + depth to 3D point (X,Y,Z)
        
        Pinhole camera equations:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth
        
        Args:
            u, v: Pixel coordinates
            depth: Depth in meters
        
        Returns:
            (X, Y, Z) in meters
        """
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z

def get_3d_bbox(bbox_2d, depth_map, camera):
    """
    Convert 2D bounding box to 3D position
    
    Args:
        bbox_2d: [x1, y1, x2, y2] in pixels
        depth_map: 2D array of depth values (uint16, millimeters)
        camera: CameraIntrinsics object
    
    Returns:
        (X, Y, Z): 3D center position in meters
        median_depth: Median depth in meters
    """
    x1, y1, x2, y2 = map(int, bbox_2d)
    
    # Extract depth values inside bounding box
    depth_roi = depth_map[y1:y2, x1:x2]
    
    # Use median (more robust than mean against outliers)
    median_depth_mm = np.median(depth_roi)
    median_depth_m = median_depth_mm / 1000.0  # Convert mm to meters
    
    # Get center pixel of box
    center_u = (x1 + x2) / 2
    center_v = (y1 + y2) / 2
    
    # Project to 3D
    X, Y, Z = camera.pixel_to_3d(center_u, center_v, median_depth_m)
    
    return X, Y, Z, median_depth_m

def check_depth_consistency(bbox_2d, depth_map, threshold=0.5):
    """
    Check if depth inside bounding box is consistent
    Filters false positives (e.g., detecting background)
    
    Args:
        bbox_2d: [x1, y1, x2, y2]
        depth_map: Depth in mm
        threshold: Max std deviation in meters
    
    Returns:
        is_valid: True if depth is consistent
        std_depth: Standard deviation
    """
    x1, y1, x2, y2 = map(int, bbox_2d)
    depth_roi = depth_map[y1:y2, x1:x2] / 1000.0  # Convert to meters
    
    std_depth = np.std(depth_roi)
    is_valid = std_depth < threshold
    
    return is_valid, std_depth

def fuse_2d_3d(detections_2d, depth_map, camera):
    """
    Fuse 2D detections with depth to get 3D positions
    
    Args:
        detections_2d: List of detection dicts with 'bbox' and 'confidence'
        depth_map: Depth map (uint16, mm)
        camera: CameraIntrinsics
    
    Returns:
        detections_3d: List with added 3D info
    """
    detections_3d = []
    
    for det in detections_2d:
        bbox = det['bbox']
        conf = det['confidence']
        
        # Get 3D position
        X, Y, Z, depth_m = get_3d_bbox(bbox, depth_map, camera)
        
        # Check depth consistency
        is_valid, std_depth = check_depth_consistency(bbox, depth_map)
        
        detections_3d.append({
            'bbox_2d': bbox,
            'confidence': conf,
            'position_3d': (X, Y, Z),
            'depth_m': depth_m,
            'depth_consistent': is_valid,
            'depth_std': std_depth
        })
    
    return detections_3d

if __name__ == "__main__":
    # Test 3D fusion
    print("=== 3D Fusion Test ===\n")
    
    from detection import load_yolo_model, detect_persons
    
    # Load test data
    rgb_path = 'dataset/rgb_normal.jpg'
    depth_path = 'dataset/depth_normal.npy'
    
    rgb = cv2.imread(rgb_path)
    depth = np.load(depth_path)
    
    print(f"Image shape: {rgb.shape}")
    print(f"Depth shape: {depth.shape}\n")
    
    # Detect persons
    model = load_yolo_model('n')
    detections_2d, _ = detect_persons(model, rgb_path)
    
    print(f"2D Detections: {len(detections_2d)}\n")
    
    # Initialize camera (adjusted for small test image)
    camera = CameraIntrinsics(fx=525.0, fy=525.0, cx=112.0, cy=74.0)
    
    # Fuse 2D + depth → 3D
    detections_3d = fuse_2d_3d(detections_2d, depth, camera)
    
    # Print results
    for i, det in enumerate(detections_3d, 1):
        print(f"Person {i}:")
        print(f"  2D Box: {det['bbox_2d']}")
        print(f"  3D Position: X={det['position_3d'][0]:.2f}m, Y={det['position_3d'][1]:.2f}m, Z={det['position_3d'][2]:.2f}m")
        print(f"  Depth: {det['depth_m']:.2f}m")
        print(f"  Depth Consistent: {det['depth_consistent']} (std={det['depth_std']:.3f}m)")
        print(f"  Confidence: {det['confidence']:.2%}\n")
    
    print("✓ 3D fusion working!")