"""
Dependability Cage Architecture
Implements the IKS safety pattern: Complex AI (Doer) + Simple Monitor (Checker)
"""

import cv2
import numpy as np
from detection import load_yolo_model, detect_persons
from fusion_3d import CameraIntrinsics, fuse_2d_3d, check_depth_consistency
from uncertainty import detect_with_uncertainty

class SafetyMonitor:
    """
    The "Checker" - Simple, verifiable safety rules
    Monitors the complex AI and triggers interventions
    """
    
    def __init__(self, uncertainty_threshold=0.3, depth_std_threshold=0.5):
        self.uncertainty_threshold = uncertainty_threshold
        self.depth_std_threshold = depth_std_threshold
        self.safety_log = []
    
    def check_detection_safety(self, detection_3d, uncertainty_score):
        """
        Validate a single detection for safety concerns
        
        Returns:
            is_safe: bool
            concerns: list of safety issues
        """
        concerns = []
        
        # Check 1: High epistemic uncertainty (unknown unknown)
        if uncertainty_score > self.uncertainty_threshold:
            concerns.append(f"HIGH_UNCERTAINTY ({uncertainty_score:.2%})")
        
        # Check 2: Depth consistency (geometric validation)
        if not detection_3d['depth_consistent']:
            concerns.append(f"DEPTH_INCONSISTENT (std={detection_3d['depth_std']:.3f}m)")
        
        # Check 3: Confidence threshold
        if detection_3d['confidence'] < 0.7:
            concerns.append(f"LOW_CONFIDENCE ({detection_3d['confidence']:.2%})")
        
        # Check 4: Physical plausibility (not too close/far)
        depth = detection_3d['depth_m']
        if depth < 0.5 or depth > 10.0:
            concerns.append(f"IMPLAUSIBLE_DEPTH ({depth:.2f}m)")
        
        is_safe = len(concerns) == 0
        
        # Log for analysis
        self.safety_log.append({
            'detection': detection_3d,
            'uncertainty': uncertainty_score,
            'is_safe': is_safe,
            'concerns': concerns
        })
        
        return is_safe, concerns
    
    def trigger_safety_intervention(self, detection, concerns):
        """
        What to do when safety monitor flags a concern
        """
        print(f"  ⚠️  SAFETY INTERVENTION TRIGGERED:")
        print(f"      Detection: {detection['bbox_2d']}")
        print(f"      Concerns: {', '.join(concerns)}")
        print(f"      Action: Flagging for geometric fallback verification")


class DependabilityCage:
    """
    The complete safety architecture:
    - Doer: YOLOv8 (complex, high-performance AI)
    - Checker: SafetyMonitor (simple, verifiable rules)
    """
    
    def __init__(self):
        print("Initializing Dependability Cage...")
        
        # The Doer (Complex AI)
        self.doer = load_yolo_model('n')
        print("  ✓ Doer loaded: YOLOv8 (complex perception)")
        
        # The Checker (Safety Monitor)
        self.checker = SafetyMonitor(
            uncertainty_threshold=0.3,
            depth_std_threshold=0.5
        )
        print("  ✓ Checker loaded: Safety Monitor (verifiable rules)")
        
        # Camera parameters
        self.camera = CameraIntrinsics(fx=525.0, fy=525.0, cx=319.5, cy=239.5)
    
    def safe_detect(self, rgb, depth_map):
        """
        Run detection with safety monitoring
        
        Args:
            rgb: RGB image
            depth_map: Depth map in meters
        
        Returns:
            safe_detections: Only validated detections
            flagged_detections: Detections with safety concerns
        """
        # Step 1: Doer runs perception
        detections_with_unc, overall_unc = detect_with_uncertainty(
            self.doer, rgb, n_augments=5
        )
        
        if len(detections_with_unc) == 0:
            return [], []
        
        # Step 2: Fuse with depth for 3D
        detections_2d = [{'bbox': d['bbox'], 'confidence': d['confidence']} 
                         for d in detections_with_unc]
        
        depth_mm = (depth_map * 1000).astype(np.uint16)  # Convert to mm
        detections_3d = fuse_2d_3d(detections_2d, depth_mm, self.camera)
        
        # Step 3: Checker validates each detection
        safe_detections = []
        flagged_detections = []
        
        for i, (det_3d, det_unc) in enumerate(zip(detections_3d, detections_with_unc)):
            is_safe, concerns = self.checker.check_detection_safety(
                det_3d, 
                det_unc['uncertainty']
            )
            
            if is_safe:
                safe_detections.append(det_3d)
            else:
                flagged_detections.append({
                    'detection': det_3d,
                    'concerns': concerns,
                    'uncertainty': det_unc['uncertainty']
                })
                self.checker.trigger_safety_intervention(det_3d, concerns)
        
        return safe_detections, flagged_detections
    
    def get_safety_statistics(self):
        """Generate safety monitoring report"""
        total = len(self.checker.safety_log)
        if total == 0:
            return "No detections processed yet."
        
        safe_count = sum(1 for log in self.checker.safety_log if log['is_safe'])
        flagged_count = total - safe_count
        
        concern_types = {}
        for log in self.checker.safety_log:
            for concern in log['concerns']:
                concern_type = concern.split('(')[0].strip()
                concern_types[concern_type] = concern_types.get(concern_type, 0) + 1
        
        report = f"""
=== SAFETY MONITORING REPORT ===
Total Detections Processed: {total}
Safe Detections: {safe_count} ({safe_count/total*100:.1f}%)
Flagged Detections: {flagged_count} ({flagged_count/total*100:.1f}%)

Safety Concerns Breakdown:
"""
        for concern, count in sorted(concern_types.items(), key=lambda x: -x[1]):
            report += f"  - {concern}: {count} occurrences\n"
        
        return report


if __name__ == "__main__":
    print("=== DEPENDABILITY CAGE TEST ===\n")
    
    # Initialize the cage
    cage = DependabilityCage()
    
    # Test on TUM frames
    dataset_path = 'TUM_Dataset/rgbd_dataset_freiburg3_walking_xyz'
    
    print("\nTesting on 3 TUM frames...\n")
    
    for frame_idx in [0, 100, 200]:
        print(f"Frame {frame_idx}:")
        
        # Load RGB-D
        from analyze_tum import load_tum_frame
        rgb, depth = load_tum_frame(dataset_path, frame_idx)
        
        # Run safe detection
        safe_dets, flagged_dets = cage.safe_detect(rgb, depth)
        
        print(f"  Safe detections: {len(safe_dets)}")
        print(f"  Flagged detections: {len(flagged_dets)}")
        print()
    
    # Print safety statistics
    print(cage.get_safety_statistics())
    
    print("\n✓ Dependability Cage architecture working!")
    print("\nThis implements the IKS 'Safe Intelligence' pattern:")
    print("  - Doer: Complex AI (YOLOv8) for high performance")
    print("  - Checker: Simple safety rules for verification")
    print("  - Result: Intelligence with architectural safety guarantees")