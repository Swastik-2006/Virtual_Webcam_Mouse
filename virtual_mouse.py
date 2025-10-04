import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ------------------ Initialization ------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

screen_w, screen_h = pyautogui.size()

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# ------------------ Variables ------------------
smooth_x, smooth_y = 0, 0
last_click_time = 0
prev_y = None
frame_count = 0

# ------------------ Main Loop ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue  # skip every other frame to reduce lag

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            # ------------------ Landmarks ------------------
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            thumb_tip = hand_landmarks.landmark[4]

            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # ------------------ Smooth Cursor ------------------
            target_x = screen_w * index_tip.x
            target_y = screen_h * index_tip.y
            smooth_x = smooth_x + (target_x - smooth_x) * 0.2
            smooth_y = smooth_y + (target_y - smooth_y) * 0.2
            pyautogui.moveTo(smooth_x, smooth_y)

            # ------------------ Gestures ------------------
            # Left Click / Double Click
            if distance((ix, iy), (tx, ty)) < 40:
                current_time = time.time()
                if current_time - last_click_time < 0.4:
                    pyautogui.doubleClick()
                    cv2.circle(frame, (ix, iy), 30, (0, 0, 255), 3)
                    last_click_time = 0
                else:
                    pyautogui.click()
                    cv2.circle(frame, (ix, iy), 20, (0, 255, 0), 3)
                    last_click_time = current_time
                pyautogui.sleep(0.2)

            # Right Click
            elif distance((mx, my), (tx, ty)) < 40:
                pyautogui.click(button="right")
                cv2.circle(frame, (mx, my), 20, (255, 0, 0), 3)
                pyautogui.sleep(0.2)

            # Scroll (index + middle close)
            if distance((ix, iy), (mx, my)) < 40:
                if prev_y is not None:
                    dy = iy - prev_y
                    if dy > 10:
                        pyautogui.scroll(-50)
                        cv2.putText(frame, "Scroll Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    elif dy < -10:
                        pyautogui.scroll(50)
                        cv2.putText(frame, "Scroll Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                prev_y = iy
            else:
                prev_y = None

    # ------------------ Display ------------------
    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
# to activate virtual environment run this on terminal→ .\venv\Scripts\activate
