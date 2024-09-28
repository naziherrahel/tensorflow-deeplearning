import cv2
import tensorflow as tf
import numpy as np
import time
import os 

# Load your class names
class_names = ["apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"]

# Load the fine-tuned EfficientNetB0 model
model = tf.keras.models.load_model('model_1.keras')

# Directory to save frames with detected objects
output_dir = "classified_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Confidence threshold for saving frames and showing labels
confidence_threshold = 0.5  # Only save frames with a confidence score above this value

# Function to preprocess input frame
def prepare_frame(frame):
    img = cv2.resize(frame, (224, 224))  # Resize frame to model input size
    img_array = np.expand_dims(img, axis=0)
    img_array = tf.keras.applications.imagenet_utils.preprocess_input(img_array)  # Preprocess for EfficientNet
    return img_array

# Custom function to decode predictions for Food101
def decode_predictions_food(preds, top=3):
    top_indices = preds[0].argsort()[-top:][::-1]
    result = [(class_names[i], preds[0][i]) for i in top_indices]
    return result

# Initialize the video stream (use 0 for webcam, or replace with video path)
video_stream = cv2.VideoCapture(0)

# Check if the video stream is opened
if not video_stream.isOpened():
    print("Error: Unable to open video stream")
    exit()

# Variables to control frame labeling and saving
frame_count = 0
fps_interval = 30  # Classify every 30 frames
start_time = time.time()
label_display_time = 60  # Number of frames to keep label displayed
current_label = ""
label_frame_counter = 0

while True:
    ret, frame = video_stream.read()  # Read frame from video
    if not ret:
        break
    
    frame_count += 1
    
    # Only run the model every `fps_interval` frames
    if frame_count % fps_interval == 0:
        # Prepare frame for prediction
        prepared_frame = prepare_frame(frame)
        
        # Run prediction
        preds = model.predict(prepared_frame)
        top_pred = decode_predictions_food(preds, top=1)[0]  # Get top prediction
        pred_class, pred_confidence = top_pred[0], top_pred[1]
        
        # Check if the confidence score is above the threshold
        if pred_confidence >= confidence_threshold:
            # Update label and reset the label frame counter
            current_label = f"{pred_class}: {pred_confidence:.4f}"
            label_frame_counter = 0  # Reset counter when a new label is predicted
            
            # Save the frame with label
            output_path = os.path.join(output_dir, f"classified_frame_{frame_count}.jpg")
            frame_with_label = frame.copy()
            cv2.putText(frame_with_label, current_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(output_path, frame_with_label)

    # Display the label for a set number of frames (even if no new prediction is made)
    if label_frame_counter < label_display_time and current_label != "":
        cv2.putText(frame, current_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        label_frame_counter += 1
    
    # Display FPS on the frame
    fps = frame_count / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show the frame with prediction
    cv2.imshow("Food Classification Stream", frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_stream.release()
cv2.destroyAllWindows()