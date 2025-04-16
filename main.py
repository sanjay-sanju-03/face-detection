import cv2
import os
import numpy as np
import mediapipe as mp
from playsound import playsound
import time

# --- Constants ---
DATA_PATH = "stall_faces"  # Directory to store registered faces
MODEL_PATH = "trainer.yml"  # Path to save trained model
THRESHOLD_CONFIDENCE = 0.6  # Confidence threshold for Mediapipe face detection
RECOGNITION_THRESHOLD = 50  # Threshold for face recognition confidence
ALERT_SOUND = "alert.mp3"  # Alert sound file
CAPTURE_DELAY = 2  # Seconds to wait between captures
TRAIN_INTERVAL = 5  # Train after every 5 new faces
MESSAGE_TIMEOUT = 3  # Seconds to show messages
FACE_SAMPLES = 3  # Number of samples to take per person

# --- Setup ---
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.7
)

# Initialize OpenCV LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
trained = os.path.exists(MODEL_PATH)  # Check if a trained model exists

if trained:
    recognizer.read(MODEL_PATH)

# Function to get all images and labels for training
def get_images_and_labels():
    face_samples = []
    ids = []
    id_count = 0
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                face_samples.append(img)
                ids.append(id_count)
                id_count += 1
    return face_samples, ids

# Function to play alert sound
def play_alert():
    if os.path.exists(ALERT_SOUND):
        try:
            playsound(ALERT_SOUND)
        except Exception as e:
            print(f"Error playing alert sound: {e}")

# Function to capture face samples
def capture_face_samples(frame, face_region, samples_needed=FACE_SAMPLES):
    samples = []
    for _ in range(samples_needed):
        # Convert face to grayscale and resize
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (200, 200))
        samples.append(resized_face)
        
        # Show progress
        cv2.putText(frame, f"Capturing: {len(samples)}/{FACE_SAMPLES}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Stall Entry System", frame)
        cv2.waitKey(500)  # Wait 500ms between samples
    
    return samples

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

print("🎥 Camera initialized. Press 'q' to quit.")

# Main loop variables
new_faces_count = 0
last_capture_time = 0
showing_message = False
message_start_time = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    current_time = time.time()

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w_box, h_box = (
                int(bboxC.xmin * w),
                int(bboxC.ymin * h),
                int(bboxC.width * w),
                int(bboxC.height * h),
            )
            x, y = max(0, x), max(0, y)  # Ensure coordinates are within bounds
            face = frame[y : y + h_box, x : x + w_box]

            # Convert face to grayscale and resize
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (200, 200))

            if trained:
                # Recognize the face
                id_, conf = recognizer.predict(resized_face)
                if conf < RECOGNITION_THRESHOLD:
                    cv2.putText(frame, "DUPLICATE ENTRY DETECTED!", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    play_alert()
                    showing_message = True
                    message_start_time = current_time
                    time.sleep(2)  # Pause to show alert
                elif current_time - last_capture_time > CAPTURE_DELAY:
                    # Capture new face
                    samples = capture_face_samples(frame, face)
                    for idx, sample in enumerate(samples):
                        new_id = len(os.listdir(DATA_PATH))
                        img_path = os.path.join(DATA_PATH, f"{new_id}_face_{idx}.jpg")
                        cv2.imwrite(img_path, sample)
                    
                    new_faces_count += 1
                    last_capture_time = current_time
                    
                    # Show "Next Person" message
                    cv2.putText(frame, "NEXT PERSON PLEASE!", 
                               (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    showing_message = True
                    message_start_time = current_time

            else:
                # First-time setup
                samples = capture_face_samples(frame, face)
                for idx, sample in enumerate(samples):
                    new_id = len(os.listdir(DATA_PATH))
                    img_path = os.path.join(DATA_PATH, f"{new_id}_face_{idx}.jpg")
                    cv2.imwrite(img_path, sample)
                
                new_faces_count += 1
                last_capture_time = current_time

            # Auto-train after collecting enough new faces
            if new_faces_count >= TRAIN_INTERVAL:
                faces, ids = get_images_and_labels()
                if faces and ids:
                    recognizer.train(faces, np.array(ids))
                    recognizer.write(MODEL_PATH)
                    trained = True
                    new_faces_count = 0
                    print("Model automatically trained and saved.")

    # Clear messages after timeout
    if showing_message and current_time - message_start_time > MESSAGE_TIMEOUT:
        showing_message = False

    # Display the frame
    cv2.imshow("Stall Entry System", frame)

    # Handle key presses
    key = cv2.waitKey(1)
    if key == ord("t"):  # Train the model
        faces, ids = get_images_and_labels()
        if faces and ids:
            recognizer.train(faces, np.array(ids))
            recognizer.write(MODEL_PATH)
            trained = True
            print("Model trained and saved.")
    elif key == ord("q"):  # Quit the program
        break

# Release resources
cap.release()
cv2.destroyAllWindows()