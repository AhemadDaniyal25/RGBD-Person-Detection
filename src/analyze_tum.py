"""
TUM RGB-D Analysis Pipeline
Tests YOLO + 3D fusion on real dynamic sequences
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from detection import load_yolo_model, detect_persons
from fusion_3d import CameraIntrinsics, fuse_2d_3d
from evaluation import evaluate_detections

def load_tum_frame(dataset_path, frame_index):
    """Load RGB-D pair from TUM dataset"""
    assoc_file = os.path.join(dataset_path, 'associate.txt')
    
    with open(assoc_file, 'r') as f:
        lines = f.readlines()
    
    line = lines[frame_index].strip().split()
    rgb_path = os.path.join(dataset_path, line[1])
    depth_path = os.path.join(dataset_path, line[3])
    
    rgb = cv2.imread(rgb_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_m = depth_raw.astype(np.float32) / 5000.0  # Convert to meters
    
    return rgb, depth_m

def analyze_tum_sequence(dataset_path, frame_indices, output_dir='outputs/tum_results'):
    """
    Analyze YOLO performance on TUM dynamic sequence
    
    Args:
        dataset_path: Path to TUM dataset
        frame_indices: List of frame numbers to analyze
        output_dir: Where to save results
    """
    print("=== TUM RGB-D DEPENDABILITY ANALYSIS ===\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model
    model = load_yolo_model('n')
    
    # TUM camera intrinsics (Kinect v1)
    camera = CameraIntrinsics(
        fx=525.0,  # Focal length X
        fy=525.0,  # Focal length Y
        cx=319.5,  # Principal point X (640/2)
        cy=239.5   # Principal point Y (480/2)
    )
    
    results = []
    
    for idx in frame_indices:
        print(f"Processing frame {idx}...")
        
        # Load data
        rgb, depth = load_tum_frame(dataset_path, idx)
        
        # Detect persons (2D)
        detections_2d, annotated = detect_persons(model, rgb, conf_threshold=0.5)
        
        print(f"  Detected: {len(detections_2d)} person(s)")
        
        # Fuse with depth (3D)
        detections_3d = fuse_2d_3d(detections_2d, depth * 1000, camera)  # Convert m to mm
        
        # Analyze safety-critical detections (< 3m)
        hazardous_detections = [d for d in detections_3d if d['depth_m'] < 3.0]
        
        print(f"  Hazardous zone (<3m): {len(hazardous_detections)} person(s)")
        
        # Save annotated frame
        cv2.imwrite(f'{output_dir}/frame_{idx:04d}_annotated.jpg', annotated)
        
        results.append({
            'frame': idx,
            'detections_2d': len(detections_2d),
            'detections_3d': detections_3d,
            'hazardous_count': len(hazardous_detections)
        })
        print()
    
    # Generate summary
    generate_tum_report(results, output_dir)
    
    return results

def generate_tum_report(results, output_dir):
    """Generate analysis report"""
    
    print("\n" + "="*60)
    print("TUM ANALYSIS SUMMARY")
    print("="*60)
    
    total_detections = sum(r['detections_2d'] for r in results)
    total_hazardous = sum(r['hazardous_count'] for r in results)
    
    print(f"\nFrames analyzed: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"Hazardous detections (<3m): {total_hazardous}")
    print(f"Hazard rate: {total_hazardous/total_detections*100:.1f}%")
    
    # Plot detection timeline
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    frames = [r['frame'] for r in results]
    detections = [r['detections_2d'] for r in results]
    hazardous = [r['hazardous_count'] for r in results]
    
    # Detection count over time
    ax1.plot(frames, detections, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax1.set_ylabel('Number of Detections', fontsize=12)
    ax1.set_title('YOLO Person Detection on TUM Dynamic Sequence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Hazardous detections
    ax2.bar(frames, hazardous, color='coral', alpha=0.8)
    ax2.set_xlabel('Frame Index', fontsize=12)
    ax2.set_ylabel('Hazardous Detections (<3m)', fontsize=12)
    ax2.set_title('Safety-Critical Detection Count', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Safe')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tum_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Analysis plot saved: {output_dir}/tum_analysis.png")

if __name__ == "__main__":
    dataset_path = 'TUM_Dataset/rgbd_dataset_freiburg3_walking_xyz'
    
    # Analyze every 50th frame (827 total, so ~16 frames)
    frame_indices = list(range(0, 827, 50))
    
    results = analyze_tum_sequence(dataset_path, frame_indices)
    
    print("\n✓ TUM analysis complete!")
    print("Next: Add uncertainty quantification and Dependability Cage architecture")