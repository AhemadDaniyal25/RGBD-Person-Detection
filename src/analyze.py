"""
Complete Analysis Pipeline
Tests YOLO under different conditions and measures degradation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from detection import load_yolo_model, detect_persons
from fusion_3d import CameraIntrinsics, fuse_2d_3d
from evaluation import evaluate_detections
import os

def run_detection_pipeline(rgb_path, depth_path, ground_truth_boxes, camera):
    """
    Run complete pipeline: detect → fuse → evaluate
    
    Args:
        rgb_path: Path to RGB image
        depth_path: Path to depth map (.npy)
        ground_truth_boxes: List of GT boxes [[x1,y1,x2,y2], ...]
        camera: CameraIntrinsics object
    
    Returns:
        results: Dict with detections, metrics, 3D info
    """
    # Load data
    rgb = cv2.imread(rgb_path)
    depth = np.load(depth_path)
    
    # Load model (cached after first load)
    model = load_yolo_model('n')
    
    # Run 2D detection
    detections_2d, annotated = detect_persons(model, rgb_path, conf_threshold=0.5)
    
    # Extract just bounding boxes for evaluation
    pred_boxes = [det['bbox'] for det in detections_2d]
    
    # Evaluate 2D detections
    metrics = evaluate_detections(pred_boxes, ground_truth_boxes, iou_threshold=0.5)
    
    # Fuse with depth for 3D
    detections_3d = fuse_2d_3d(detections_2d, depth, camera)
    
    return {
        'detections_2d': detections_2d,
        'detections_3d': detections_3d,
        'metrics': metrics,
        'annotated': annotated
    }

def analyze_all_conditions(dataset_dir='dataset', output_dir='outputs'):
    """
    Test YOLO across all conditions and measure degradation
    
    Args:
        dataset_dir: Where synthetic data is stored
        output_dir: Where to save results
    
    Returns:
        all_results: Dict of results per condition
    """
    print("=== DEPENDABILITY ANALYSIS ===\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Ground truth from normal condition
    print("Establishing ground truth...")
    model = load_yolo_model('n')
    gt_detections, _ = detect_persons(model, f'{dataset_dir}/rgb_normal.jpg', conf_threshold=0.5)
    ground_truth = [det['bbox'] for det in gt_detections]
    
    print(f"Ground truth: {len(ground_truth)} person(s)\n")
    
    # Initialize camera (adjusted for test image size)
    camera = CameraIntrinsics(fx=525.0, fy=525.0, cx=112.0, cy=74.0)
    
    # Test conditions
    conditions = {
        'Normal': ('rgb_normal.jpg', 'depth_normal.npy'),
        '25% Occluded': ('rgb_occ25.jpg', 'depth_occ25.npy'),
        '50% Occluded': ('rgb_occ50.jpg', 'depth_occ50.npy'),
        'Bright Light': ('rgb_bright.jpg', 'depth_bright.npy'),
        'Dark Light': ('rgb_dark.jpg', 'depth_dark.npy'),
    }
    
    all_results = {}
    
    for condition, (rgb_file, depth_file) in conditions.items():
        print(f"Testing: {condition}")
        
        rgb_path = f'{dataset_dir}/{rgb_file}'
        depth_path = f'{dataset_dir}/{depth_file}'
        
        result = run_detection_pipeline(rgb_path, depth_path, ground_truth, camera)
        all_results[condition] = result
        
        # Print metrics
        metrics = result['metrics']
        print(f"  Detections: {metrics['num_predictions']}/{metrics['num_ground_truth']}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  Precision: {metrics['precision']:.2%}")
        
        if metrics['avg_iou'] > 0:
            print(f"  Avg IoU: {metrics['avg_iou']:.3f}")
        
        # Save annotated image
        cv2.imwrite(f'{output_dir}/result_{condition.replace(" ", "_").lower()}.jpg', 
                   result['annotated'])
        print()
    
    # Generate performance plots
    plot_performance_degradation(all_results, output_dir)
    
    return all_results

def plot_performance_degradation(results, output_dir):
    """
    Create visualization of performance across conditions
    """
    conditions = list(results.keys())
    recalls = [results[c]['metrics']['recall'] for c in conditions]
    precisions = [results[c]['metrics']['precision'] for c in conditions]
    num_detections = [results[c]['metrics']['num_predictions'] for c in conditions]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Recall degradation
    axes[0].bar(conditions, recalls, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Recall', fontsize=12)
    axes[0].set_title('Recall Degradation Across Conditions', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
    axes[0].axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Acceptable')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend()
    for i, v in enumerate(recalls):
        axes[0].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')
    
    # Precision
    axes[1].bar(conditions, precisions, color='coral', alpha=0.8)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision Across Conditions', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(precisions):
        axes[1].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')
    
    # Number of detections
    axes[2].bar(conditions, num_detections, color='mediumpurple', alpha=0.8)
    axes[2].set_ylabel('Number of Detections', fontsize=12)
    axes[2].set_title('Detection Count per Condition', fontsize=14, fontweight='bold')
    axes[2].axhline(y=results['Normal']['metrics']['num_ground_truth'], 
                    color='green', linestyle='--', alpha=0.5, label='Ground Truth')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].legend()
    for i, v in enumerate(num_detections):
        axes[2].text(i, v + 0.05, str(v), ha='center', fontweight='bold')
    
    # Rotate x-axis labels
    for ax in axes:
        ax.set_xticklabels(conditions, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_degradation.png', dpi=150, bbox_inches='tight')
    print(f"✓ Performance plot saved: {output_dir}/performance_degradation.png")

def generate_report(results, output_dir='outputs'):
    """
    Generate text report with key findings
    """
    report_path = f'{output_dir}/analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("DEPENDABILITY ANALYSIS REPORT\n")
        f.write("RGB-D Person Detection Performance Under Stress\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 60 + "\n")
        
        for condition, result in results.items():
            metrics = result['metrics']
            f.write(f"\n{condition}:\n")
            f.write(f"  Recall:     {metrics['recall']:.2%}\n")
            f.write(f"  Precision:  {metrics['precision']:.2%}\n")
            f.write(f"  Detections: {metrics['num_predictions']}/{metrics['num_ground_truth']}\n")
            
            if metrics['avg_iou'] > 0:
                f.write(f"  Avg IoU:    {metrics['avg_iou']:.3f}\n")
        
        # Key findings
        f.write("\n" + "=" * 60 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("-" * 60 + "\n")
        
        normal_recall = results['Normal']['metrics']['recall']
        occ50_recall = results['50% Occluded']['metrics']['recall']
        
        recall_drop = (normal_recall - occ50_recall) * 100
        
        f.write(f"\n1. OCCLUSION IMPACT:\n")
        f.write(f"   - Baseline recall: {normal_recall:.0%}\n")
        f.write(f"   - 50% occlusion recall: {occ50_recall:.0%}\n")
        f.write(f"   - Performance drop: {recall_drop:.0f} percentage points\n")
        
        if occ50_recall < 0.8:
            f.write(f"   ⚠ WARNING: Recall below safety threshold (80%)\n")
        
        f.write(f"\n2. LIGHTING ROBUSTNESS:\n")
        bright_recall = results['Bright Light']['metrics']['recall']
        dark_recall = results['Dark Light']['metrics']['recall']
        f.write(f"   - Bright lighting: {bright_recall:.0%}\n")
        f.write(f"   - Dark lighting: {dark_recall:.0%}\n")
        
        f.write(f"\n3. RECOMMENDATIONS:\n")
        if occ50_recall < 0.8:
            f.write(f"   - Implement multi-view camera setup for occlusion handling\n")
        if dark_recall < 0.8:
            f.write(f"   - Add IR/thermal cameras for low-light scenarios\n")
        f.write(f"   - Use depth consistency filtering (already implemented)\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✓ Report saved: {report_path}")

if __name__ == "__main__":
    # Run complete analysis
    results = analyze_all_conditions()
    
    # Generate report
    generate_report(results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated outputs:")
    print("  📊 outputs/performance_degradation.png")
    print("  📝 outputs/analysis_report.txt")
    print("  🖼️  outputs/result_*.jpg (annotated images)")
    print("\n✓ Project complete! Ready for interview demo.")