import cv2
import time 
import math
import cvzone
import numpy as np
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

# x is the value of the distace that we are getting 
# y is the distance from the camera to the hand we get in cm
x = [300, 220, 190, 150, 135, 115, 105, 85]
y = [20, 25, 30, 35, 40, 50, 60, 70]

# As the relation is not linear so we have to make a polinomial equation

#   Like Y = Ax^2 + Bx + C
#   So for every x value it give us the value of y that matches the above tabel

# Here comes the numpy function polyfit (x, y, degree)

coff = np.polyfit(x, y, 2)


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

            distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)) 

            A, B, C = coff

            distanceCM = int(A*distance**2 + B*distance + C)
            # print(distance, distanceCM)

            for start, end in HAND_CONNECTIONS:
                cv2.line(img, lm_list[start], lm_list[end], (0, 255, 0), 2)
            
            for x, y in lm_list:
                cv2.circle(img, (x, y), 6, (255, 0, 0), -1)


            cvzone.putTextRect(img,
                               f"{distanceCM}cm",
                               (x_min-20, y_min-35),
                               )
            cv2.rectangle(
                img,
                (x_min-20, y_min-20),
                (x_max+20, y_max+20),
                (0, 255, 0),
                2
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