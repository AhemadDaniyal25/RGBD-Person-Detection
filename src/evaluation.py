"""
Evaluation Metrics Module
Calculates IoU, Precision, Recall for object detection
"""

import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU)
    
    Args:
        box1, box2: [x1, y1, x2, y2]
    
    Returns:
        iou: Float between 0 and 1
    """
    # Get intersection rectangle
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    
    # Calculate intersection area
    if x2_i < x1_i or y2_i < y1_i:
        intersection = 0.0
    else:
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    return iou

def evaluate_detections(predictions, ground_truth, iou_threshold=0.5):
    """
    Evaluate predictions against ground truth
    
    Args:
        predictions: List of predicted boxes [x1,y1,x2,y2]
        ground_truth: List of ground truth boxes
        iou_threshold: Minimum IoU to count as match
    
    Returns:
        metrics: Dict with precision, recall, TP, FP, FN
    """
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    matched_gt = set()
    matched_pred_ious = []
    
    # For each prediction, find best matching ground truth
    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        
        for i, gt in enumerate(ground_truth):
            if i not in matched_gt:
                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        # Check if match is good enough
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
            matched_pred_ious.append(best_iou)
        else:
            fp += 1
    
    # Unmatched ground truth = missed detections
    fn = len(ground_truth) - len(matched_gt)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Average IoU of matched detections
    avg_iou = np.mean(matched_pred_ious) if matched_pred_ious else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'avg_iou': avg_iou,
        'num_predictions': len(predictions),
        'num_ground_truth': len(ground_truth)
    }

if __name__ == "__main__":
    # Test metrics
    print("=== Evaluation Metrics Test ===\n")
    
    # Example 1: Perfect detection
    print("Test 1: Perfect detection")
    pred = [[100, 100, 200, 300]]
    gt = [[100, 100, 200, 300]]
    iou = calculate_iou(pred[0], gt[0])
    print(f"  IoU: {iou:.3f} (Expected: 1.0)\n")
    
    # Example 2: Partial overlap
    print("Test 2: Partial overlap")
    pred = [[100, 100, 200, 300]]
    gt = [[110, 110, 210, 310]]
    iou = calculate_iou(pred[0], gt[0])
    print(f"  IoU: {iou:.3f}\n")
    
    # Example 3: No overlap
    print("Test 3: No overlap")
    pred = [[100, 100, 200, 300]]
    gt = [[300, 300, 400, 500]]
    iou = calculate_iou(pred[0], gt[0])
    print(f"  IoU: {iou:.3f} (Expected: 0.0)\n")
    
    # Example 4: Full evaluation
    print("Test 4: Complete evaluation")
    predictions = [
        [100, 100, 200, 300],  # Matches GT1
        [250, 150, 350, 400],  # Matches GT2
        [500, 500, 600, 700]   # False positive
    ]
    ground_truth = [
        [105, 105, 205, 305],  # Matches Pred1
        [245, 145, 355, 405],  # Matches Pred2
        [800, 800, 900, 1000]  # Missed (FN)
    ]
    
    metrics = evaluate_detections(predictions, ground_truth, iou_threshold=0.5)
    
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall: {metrics['recall']:.2f}")
    print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print(f"  Avg IoU: {metrics['avg_iou']:.3f}\n")
    
    print("✓ Evaluation metrics working!")