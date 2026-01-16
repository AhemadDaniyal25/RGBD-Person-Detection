"""
YOLO Detection Module
Detects persons in RGB images using pre-trained YOLOv8
"""

from ultralytics import YOLO
import cv2
import numpy as np

def load_yolo_model(model_size='n'):
    """
    Load pre-trained YOLO model
    
    Args:
        model_size: 'n' (nano/fastest), 's' (small), 'm' (medium)
    
    Returns:
        model: YOLO detector
    """
    model_name = f'yolov8{model_size}.pt'
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    return model

def detect_persons(model, image_path, conf_threshold=0.5):
    """
    Detect persons in an image
    
    Args:
        model: YOLO model
        image_path: Path to input image
        conf_threshold: Minimum confidence (0.0 to 1.0)
    
    Returns:
        detections: List of detections with bbox and confidence
        annotated_image: Image with bounding boxes drawn
    """
    # Run YOLO inference
    results = model(image_path, verbose=False)
    
    # Extract person detections (class 0 in COCO dataset)
    detections = []
    
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Filter: only persons with sufficient confidence
        if cls == 0 and conf >= conf_threshold:
            coords = box.xyxy[0].cpu().numpy()
            detections.append({
                'bbox': coords,  # [x1, y1, x2, y2]
                'confidence': conf
            })
    
    # Get annotated image
    annotated = results[0].plot()
    
    return detections, annotated

if __name__ == "__main__":
    # Test the detection module
    print("=== YOLO Person Detection Test ===\n")
    
    # Load model (downloads ~6MB on first run)
    model = load_yolo_model('n')
    
    # Run detection
    image_path = 'test_person.jpg'
    detections, annotated = detect_persons(model, image_path)
    
    # Print results
    print(f"Found {len(detections)} person(s):\n")
    for i, det in enumerate(detections, 1):
        bbox = det['bbox']
        conf = det['confidence']
        print(f"Person {i}:")
        print(f"  Bounding Box: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
        print(f"  Confidence: {conf:.2%}\n")
    
    # Save result
    cv2.imwrite('outputs/detection_result.jpg', annotated)
    print("✓ Annotated image saved to: outputs/detection_result.jpg")