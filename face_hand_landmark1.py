# Import Libraries
import cv2
import time
import mediapipe as mp

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------- GIVE YOUR VIDEO PATH HERE ----------
video_path = r"C:\Users\Admin\Desktop\MEDIAPIPE\3 face_hand_landmark\From Main Klickpin CF- Ronaldo on Saudi founding day!🇸🇦🫶🏻 - 2vczLFUdu.mp4"

# Capture video from file
capture = cv2.VideoCapture(video_path)

previousTime = 0

while capture.isOpened():
    ret, frame = capture.read()
    
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))

    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    results = holistic_model.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw Face Landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS
        )

    # Draw Right Hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # Draw Left Hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # FPS Calculation
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime) if previousTime != 0 else 0
    previousTime = currentTime

    cv2.putText(image, f"{int(fps)} FPS", (10, 70),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Facial and Hand Landmarks - Video", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()