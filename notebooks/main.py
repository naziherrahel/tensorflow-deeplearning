from ultralytics import YOLO
import cv2
import math

# Set the video source and output file paths
input_video_path = 'data_test/demo3.mp4'  # Change this to your input video path
output_video_path = 'output.avi'  # Change this to your desired output path

# Initialize video capture
cap = cv2.VideoCapture(0)


# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

# Load the YOLO model
model = YOLO("best.pt")  
classNames = ['helmet', 'not_helmet', 'not_reflective', 'reflective']

# Desired width and height for the display window
display_width = 1024  
display_height = int((display_width / frame_width) * frame_height)

while True:
    success, img = cap.read()
    if not success:
        break  # Exit loop if video ends

    # Doing detections using YOLOv8 frame by frame
    results = model(img, stream=True)

    # Loop through detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name} {conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=2, thickness=2)[0]

            # Draw label background and text
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # Filled rectangle
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    # Resize the image for display
    img_resized = cv2.resize(img, (display_width, display_height))
    # Write the output frame
    out.write(img)

    # Display the output frame
    cv2.imshow('Object Detection', img_resized)

    # Break the loop if the '1' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
