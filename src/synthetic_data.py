"""
Synthetic RGB-D Data Generator
Creates controlled test conditions: occlusion, lighting variations
"""

import cv2
import numpy as np
import os
from detection import load_yolo_model, detect_persons

def create_synthetic_depth(rgb_image, person_boxes, person_depth=2500, bg_depth=5000):
    """
    Create synthetic depth map
    
    Args:
        rgb_image: Input RGB image
        person_boxes: List of person bounding boxes
        person_depth: Depth of persons in mm (default 2.5m)
        bg_depth: Background depth in mm (default 5m)
    
    Returns:
        depth_map: Synthetic depth in uint16 (millimeters)
    """
    h, w = rgb_image.shape[:2]
    depth_map = np.full((h, w), bg_depth, dtype=np.uint16)
    
    # Set person regions to closer depth
    for det in person_boxes:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        depth_map[y1:y2, x1:x2] = person_depth
    
    return depth_map

def simulate_occlusion(rgb_image, depth_map, occlusion_level=0.5):
    """
    Simulate partial occlusion with vertical bar
    
    Args:
        rgb_image: Input RGB
        depth_map: Input depth
        occlusion_level: 0.0 (none) to 1.0 (fully blocked)
    
    Returns:
        occluded_rgb, occluded_depth
    """
    h, w = rgb_image.shape[:2]
    occluded_rgb = rgb_image.copy()
    occluded_depth = depth_map.copy()
    
    # Calculate occlusion bar width and position
    bar_width = int(w * occlusion_level)
    bar_x = w // 2 - bar_width // 2
    
    # Draw gray bar (simulating obstacle)
    occluded_rgb[:, bar_x:bar_x+bar_width] = [128, 128, 128]
    
    # Obstacle at 1 meter depth
    occluded_depth[:, bar_x:bar_x+bar_width] = 1000
    
    return occluded_rgb, occluded_depth

def simulate_lighting(rgb_image, brightness_factor=1.5):
    """
    Simulate lighting changes
    
    Args:
        brightness_factor: >1 = brighter, <1 = darker
    
    Returns:
        adjusted_rgb
    """
    adjusted = cv2.convertScaleAbs(rgb_image, alpha=brightness_factor, beta=0)
    return adjusted

def generate_test_dataset(input_image_path, output_dir='dataset'):
    """
    Generate complete test dataset with multiple conditions
    
    Args:
        input_image_path: Path to input image
        output_dir: Where to save generated data
    """
    print("=== Generating Synthetic RGB-D Dataset ===\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    rgb = cv2.imread(input_image_path)
    if rgb is None:
        raise ValueError(f"Could not load image: {input_image_path}")
    
    print(f"Input image shape: {rgb.shape}")
    
    # Detect persons to get ground truth boxes
    print("\nDetecting persons for ground truth...")
    model = load_yolo_model('n')
    detections, _ = detect_persons(model, input_image_path)
    
    if len(detections) == 0:
        print("⚠ No persons detected! Using full image as fallback.")
        h, w = rgb.shape[:2]
        detections = [{'bbox': np.array([0, 0, w, h]), 'confidence': 1.0}]
    
    print(f"Found {len(detections)} person(s)")
    
    # Generate synthetic depth
    print("\nGenerating synthetic depth map...")
    depth = create_synthetic_depth(rgb, detections)
    
    # Save test conditions
    conditions = []
    
    # 1. Normal (baseline)
    print("Creating: Normal condition")
    cv2.imwrite(f'{output_dir}/rgb_normal.jpg', rgb)
    np.save(f'{output_dir}/depth_normal.npy', depth)
    conditions.append('normal')
    
    # 2. 25% Occlusion
    print("Creating: 25% occlusion")
    rgb_occ25, depth_occ25 = simulate_occlusion(rgb, depth, 0.25)
    cv2.imwrite(f'{output_dir}/rgb_occ25.jpg', rgb_occ25)
    np.save(f'{output_dir}/depth_occ25.npy', depth_occ25)
    conditions.append('occ25')
    
    # 3. 50% Occlusion
    print("Creating: 50% occlusion")
    rgb_occ50, depth_occ50 = simulate_occlusion(rgb, depth, 0.50)
    cv2.imwrite(f'{output_dir}/rgb_occ50.jpg', rgb_occ50)
    np.save(f'{output_dir}/depth_occ50.npy', depth_occ50)
    conditions.append('occ50')
    
    # 4. Bright lighting
    print("Creating: Bright lighting")
    rgb_bright = simulate_lighting(rgb, 1.5)
    cv2.imwrite(f'{output_dir}/rgb_bright.jpg', rgb_bright)
    np.save(f'{output_dir}/depth_bright.npy', depth)
    conditions.append('bright')
    
    # 5. Dark lighting
    print("Creating: Dark lighting")
    rgb_dark = simulate_lighting(rgb, 0.5)
    cv2.imwrite(f'{output_dir}/rgb_dark.jpg', rgb_dark)
    np.save(f'{output_dir}/depth_dark.npy', depth)
    conditions.append('dark')
    
    print(f"\n✓ Dataset generated: {len(conditions)} conditions")
    print(f"  Location: {output_dir}/")
    
    return conditions, detections

if __name__ == "__main__":
    # Generate dataset from test image
    conditions, ground_truth = generate_test_dataset('test_person.jpg')
    
    print("\n=== Ground Truth ===")
    for i, det in enumerate(ground_truth, 1):
        bbox = det['bbox']
        print(f"Person {i}: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")