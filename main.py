import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.load_weights('digit_weights.weights.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def get_hand_info(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if not result.multi_hand_landmarks:
        return None, False

    hand_landmarks = result.multi_hand_landmarks[0]
    h, w, _ = frame.shape
    lm = hand_landmarks.landmark

    tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x, y = int(tip.x * w), int(tip.y * h)

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    is_drawing = index_up and not middle_up and not ring_up and not pinky_up

    return (x, y), is_drawing
def preprocess_canvas(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Find the bounding box of the drawn digit
    coords = cv2.findNonZero(gray)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    digit = gray[y:y+h, x:x+w]

    # Add padding around the digit (like MNIST images have)
    padding = 20
    digit = cv2.copyMakeBorder(digit, padding, padding, padding, padding,
                                 cv2.BORDER_CONSTANT, value=0)

    resized = cv2.resize(digit, (28, 28))
    normalized = resized.astype('float32') / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped
def predict_digit(canvas):
    processed = preprocess_canvas(canvas)
    if processed is None:
        return None, 0
    prediction = model.predict(processed, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return digit, confidence

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
canvas = np.zeros_like(frame)
prev_point = None
result_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    tip_position, is_drawing = get_hand_info(frame)

    if tip_position and is_drawing:
        if prev_point:
            distance = np.hypot(tip_position[0] - prev_point[0], tip_position[1] - prev_point[1])
            if distance < 60:
               cv2.line(canvas, prev_point, tip_position, (255, 255, 255), 8)
        prev_point = tip_position
    else:
        prev_point = None

    combined = cv2.addWeighted(frame, 0.5, canvas, 0.8, 0)
    cv2.putText(combined, result_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "c: clear | p: predict | q: quit", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Air Digit Recognition", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros_like(frame)
        result_text = ""
    elif key == ord('p'):
        digit, confidence = predict_digit(canvas)
        result_text = f"Predicted: {digit} ({confidence:.2f})"
        print(result_text)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
