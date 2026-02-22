import cv2
import time 
import math
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisualRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = "hand_landmarker.task"),
    running_mode = VisualRunningMode.IMAGE,
    num_hands = 2
)

detector = HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                  # Thumb
    (0, 5), (5, 9), (9, 13), (13, 17), (17, 0),      # Wist
    (5, 6), (6, 7), (7, 8),                          # Index
    (9, 10), (10, 11), (11, 12),                     # Middle
    (13, 14), (14, 15), (15, 16),                    # Ring
    (17, 18), (18, 19), (19, 20)                     # Pinky
]

p_time = 0

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    
    if not success:
        break
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(mp.ImageFormat.SRGB,rgb)
    
    result = detector.detect(mp_img)
    
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            h, w, _ = img.shape
            lm_list = []
            x_list = []
            y_list = []

            
            for lm in hand:
                lm_list.append((int(lm.x*w), int(lm.y*h)))
                x_list.append(int(lm.x*w))
                y_list.append(int(lm.y*h))

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            x1, y1 = lm_list[5]
            x2, y2 = lm_list[17]

            for start, end in HAND_CONNECTIONS:
                cv2.line(img, lm_list[start], lm_list[end], (0, 255, 0), 2)
            
            for x, y in lm_list:
                cv2.circle(img, (x, y), 6, (255, 0, 0), -1)

            cv2.rectangle(
                img,
                (x_max, y_min),
                (x_max, y_max),
                (0, 255, 0),
                cv2.FILLED
            )

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img,
                f'FPS: {int(fps)}',
                (10, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 0, 255),
                2
                )
                
    
    cv2.imshow("Distance Measurement", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break



cap.release()
cv2.destroyAllWindows()