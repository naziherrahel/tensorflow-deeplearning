import streamlit as st
import cv2
import tempfile
from yolo_utils import load_yolo_model, process_video_frame

# Constants
MODEL_PATH = "models/best.pt"  # Path to the YOLO model
CLASS_NAMES = ['helmet', 'not_helmet', 'not_reflective', 'reflective']
DISPLAY_WIDTH = 640  # Width of displayed frames


def main():
    
    st.title("YOLO Object Detection - Video Processing")

    # Load YOLO model
    model = load_yolo_model(MODEL_PATH)

    # Select input type: Upload a video or use Webcam
    input_type = st.radio("Select Video Source", ('Upload Video', 'Use Webcam'))

    if input_type == 'Upload Video':
        # File uploader for video files
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        
        if uploaded_video is not None:
            # Save uploaded video to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
                temp_video_file.write(uploaded_video.read())
                video_path = temp_video_file.name
            
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()  # Placeholder for video frames

            # Process each frame in the video
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # Process frame with YOLO
                processed_frame = process_video_frame(model, frame, CLASS_NAMES, DISPLAY_WIDTH)

                # Display the processed frame in Streamlit
                stframe.image(processed_frame, channels="RGB")

                # Wait for key press to control frame rate (30ms = ~33 FPS)
                if cv2.waitKey(30) & 0xFF == ord('q'):  
                    break

            # Release video capture after processing
            cap.release()

    elif input_type == 'Use Webcam':
        if 'stop_webcam' not in st.session_state:
            st.session_state.stop_webcam = False

        if st.button("Stop Webcam"):
            st.session_state.stop_webcam = True

        # Initialize webcam video capture
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            # check the webcam resolution and FPS
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            st.write(f"Webcam resolution: {width}x{height} at {fps} FPS")

            # Process frames from webcam in real-time
            stframe = st.empty()  # Placeholder for the video stream
            while cap.isOpened() and not st.session_state.stop_webcam:
                success, frame = cap.read()
                if not success:
                    break

                # Perform object detection on the frame
                processed_frame = process_video_frame(model, frame, CLASS_NAMES, DISPLAY_WIDTH)

                # Display the processed frame in Streamlit
                stframe.image(processed_frame, channels="RGB")

                # Wait for key press to control frame rate (30ms = ~33 FPS)
                if cv2.waitKey(30) & 0xFF == ord('q'):  # Optionally quit on 'q'
                    break

            # Reset the stop webcam flag and release the webcam
            st.session_state.stop_webcam = False
            cap.release()


if __name__ == "__main__":
    main()