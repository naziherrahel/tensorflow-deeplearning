import cv2
from ultralytics import YOLO

# Load YOLO model
def load_yolo_model(model_path):
    """Load YOLO model with the given path."""
    return YOLO(model_path)

# Perform YOLO detection on a video frame
def detect_objects(model, frame, class_names):
    """Detect objects on a single frame using the YOLO model."""
    results = model(frame, stream = True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(box.conf[0].item(), 2)
            cls = int(box.cls[0])
            label = f"{class_names[cls]} {conf}"
            draw_bounding_box(frame, x1, y1, x2, y2, label)
    return frame

# Draw bounding boxes and labels
def draw_bounding_box(frame, x1, y1, x2, y2, label):
    """Draw bounding box and label on the frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)



def process_video_frame(model, frame, class_names, display_width):
    """Detect objects and return the processed frame."""
    frame = detect_objects(model, frame, class_names)
    resized_frame = cv2.resize(frame, (display_width, int(display_width / frame.shape[1] * frame.shape[0])))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    return rgb_frame
