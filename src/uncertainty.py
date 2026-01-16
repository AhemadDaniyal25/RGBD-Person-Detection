"""
Uncertainty Quantification for Object Detection
Implements Test-Time Augmentation (TTA) variance analysis
"""

import cv2
import numpy as np
from detection import detect_persons

def detect_with_uncertainty(model, image, n_augments=5, conf_threshold=0.5):
    """
    Run detection with Test-Time Augmentation to estimate uncertainty
    
    Args:
        model: YOLO model
        image: Input RGB image (numpy array or path)
        n_augments: Number of augmented versions to test
        conf_threshold: Confidence threshold
    
    Returns:
        detections: List with uncertainty scores
        epistemic_uncertainty: Overall uncertainty metric
    """
    
    # Load image if path provided
    if isinstance(image, str):
        image = cv2.imread(image)
    
    h, w = image.shape[:2]
    
    # Store all predictions from augmented versions
    all_predictions = []
    
    # 1. Original image
    dets_orig, _ = detect_persons(model, image, conf_threshold)
    all_predictions.append(dets_orig)
    
    # 2. Horizontal flip
    img_flip = cv2.flip(image, 1)
    dets_flip, _ = detect_persons(model, img_flip, conf_threshold)
    # Un-flip the boxes
    dets_flip_unflipped = []
    for det in dets_flip:
        bbox = det['bbox'].copy()
        bbox[0], bbox[2] = w - bbox[2], w - bbox[0]  # Flip x coordinates
        dets_flip_unflipped.append({'bbox': bbox, 'confidence': det['confidence']})
    all_predictions.append(dets_flip_unflipped)
    
    # 3. Brightness variations
    for brightness in [0.7, 1.3]:
        img_bright = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        dets_bright, _ = detect_persons(model, img_bright, conf_threshold)
        all_predictions.append(dets_bright)
    
    # 4. Scale variations (resize + crop back)
    for scale in [0.9, 1.1]:
        img_scaled = cv2.resize(image, None, fx=scale, fy=scale)
        if scale < 1.0:
            # Pad to original size
            pad_h = h - img_scaled.shape[0]
            pad_w = w - img_scaled.shape[1]
            img_scaled = cv2.copyMakeBorder(img_scaled, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)
        else:
            # Crop to original size
            img_scaled = img_scaled[:h, :w]
        
        dets_scaled, _ = detect_persons(model, img_scaled, conf_threshold)
        all_predictions.append(dets_scaled)
    
    # Analyze consistency across augmentations
    if len(dets_orig) == 0:
        return [], 1.0  # High uncertainty if nothing detected
    
    # For each original detection, check if it appears in other augmentations
    detections_with_uncertainty = []
    
    for orig_det in dets_orig:
        orig_bbox = orig_det['bbox']
        
        # Count how many augmentations detected this object
        detection_count = 0
        iou_scores = []
        
        for aug_preds in all_predictions:
            # Check if any box in this augmentation matches
            for aug_det in aug_preds:
                iou = calculate_iou_simple(orig_bbox, aug_det['bbox'])
                if iou > 0.3:  # Loose matching threshold
                    detection_count += 1
                    iou_scores.append(iou)
                    break
        
        # Consistency ratio = how often this detection appeared
        consistency = detection_count / len(all_predictions)
        
        # Uncertainty = 1 - consistency
        uncertainty = 1.0 - consistency
        
        detections_with_uncertainty.append({
            'bbox': orig_bbox,
            'confidence': orig_det['confidence'],
            'consistency': consistency,
            'uncertainty': uncertainty,
            'detection_rate': f"{detection_count}/{len(all_predictions)}"
        })
    
    # Overall epistemic uncertainty (average across all detections)
    avg_uncertainty = np.mean([d['uncertainty'] for d in detections_with_uncertainty])
    
    return detections_with_uncertainty, avg_uncertainty

def calculate_iou_simple(box1, box2):
    """Calculate IoU between two boxes"""
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

if __name__ == "__main__":
    from detection import load_yolo_model
    import cv2
    
    print("=== UNCERTAINTY QUANTIFICATION TEST ===\n")
    
    # Load model
    model = load_yolo_model('n')
    
    # Test on TUM frame
    test_image = 'TUM_Dataset/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.592026.png'
    
    print("Running TTA uncertainty analysis...")
    detections, overall_uncertainty = detect_with_uncertainty(model, test_image, n_augments=5)
    
    print(f"\nResults:")
    print(f"  Detections: {len(detections)}")
    print(f"  Overall Epistemic Uncertainty: {overall_uncertainty:.2%}\n")
    
    for i, det in enumerate(detections, 1):
        print(f"Person {i}:")
        print(f"  Confidence: {det['confidence']:.2%}")
        print(f"  Consistency: {det['consistency']:.2%} ({det['detection_rate']})")
        print(f"  Uncertainty: {det['uncertainty']:.2%}")
        
        if det['uncertainty'] > 0.3:
            print(f"  ⚠️  HIGH UNCERTAINTY - Potential OOD or occlusion")
        print()
    
    print("✓ Uncertainty quantification working!")