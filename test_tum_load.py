"""
Test loading TUM RGB-D dataset
"""
import cv2
import numpy as np
import os

def load_tum_rgbd_pair(dataset_path, pair_index=0):
    """
    Load synchronized RGB-D pair from TUM dataset
    
    Args:
        dataset_path: Path to TUM dataset folder
        pair_index: Which pair to load (0 to 826)
    
    Returns:
        rgb, depth_meters, rgb_timestamp
    """
    # Read associations
    assoc_file = os.path.join(dataset_path, 'associate.txt')
    
    with open(assoc_file, 'r') as f:
        lines = f.readlines()
    
    # Parse line: rgb_timestamp rgb_path depth_timestamp depth_path
    line = lines[pair_index].strip().split()
    rgb_timestamp = float(line[0])
    rgb_path = line[1]
    depth_timestamp = float(line[2])
    depth_path = line[3]
    
    # Load images
    rgb_full_path = os.path.join(dataset_path, rgb_path)
    depth_full_path = os.path.join(dataset_path, depth_path)
    
    rgb = cv2.imread(rgb_full_path)
    depth_raw = cv2.imread(depth_full_path, cv2.IMREAD_UNCHANGED)  # 16-bit
    
    # TUM depth: uint16, scaled by 5000
    # Divide by 5000 to get meters
    depth_meters = depth_raw.astype(np.float32) / 5000.0
    
    print(f"Loaded pair {pair_index}:")
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Depth shape: {depth_meters.shape}")
    print(f"  Depth range: {depth_meters.min():.2f}m - {depth_meters.max():.2f}m")
    print(f"  Time sync: {abs(rgb_timestamp - depth_timestamp)*1000:.1f}ms")
    
    return rgb, depth_meters, rgb_timestamp

if __name__ == "__main__":
    dataset_path = 'TUM_Dataset/rgbd_dataset_freiburg3_walking_xyz'
    
    # Test loading first frame
    rgb, depth, timestamp = load_tum_rgbd_pair(dataset_path, pair_index=0)
    
    # Save RGB
    cv2.imwrite('outputs/tum_test_rgb.jpg', rgb)
    
    # Visualize depth (normalize for display)
    depth_vis = np.clip(depth / 5.0, 0, 1)  # Clip to 5m max
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite('outputs/tum_test_depth.jpg', depth_vis)
    
    print("\n✓ Test images saved:")
    print("  - outputs/tum_test_rgb.jpg")
    print("  - outputs/tum_test_depth.jpg")
    
    # Test loading multiple frames
    print("\nTesting multiple frames:")
    for i in [0, 100, 200, 400, 600, 800]:
        try:
            rgb, depth, _ = load_tum_rgbd_pair(dataset_path, i)
            print(f"  Frame {i}: ✓")
        except Exception as e:
            print(f"  Frame {i}: ✗ ({e})")