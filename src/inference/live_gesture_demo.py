import cv2
import mediapipe as mp
from collections import deque, Counter

from .predictor import detect_gesture_from_landmarks
from .overlay_inference import load_avatars, overlay_avatar, overlay_gesture_text


def main():
    print("▶️ Starting RT-Gesture3D demo...")

    avatars = load_avatars(size=(150, 150))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    # last few predictions ke liye (to reduce flicker)
    label_history = deque(maxlen=7)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Camera could not be opened. Check if another app is using it.")
        return

    print("✅ Camera opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to read from camera.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        h, w, _ = frame.shape
        gesture_key = "neutral"
        conf = 0.0

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # mediapipe → pixel coords list
                pts = []
                for lm in handLms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    pts.append((x, y, lm.z))

                gesture_key, gid, conf = detect_gesture_from_landmarks(pts, w, h)

        # prediction smoothing
        label_history.append(gesture_key)
        stable_label = gesture_key
        if len(label_history) > 0:
            stable_label = Counter(label_history).most_common(1)[0][0]

        # text overlay
        frame = overlay_gesture_text(frame, stable_label, conf)

        # avatar overlay
        avatar_img = avatars.get(stable_label)
        if avatar_img is not None:
            frame = overlay_avatar(frame, avatar_img)

        cv2.imshow("RT-Gesture3D - Live Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 Q pressed, exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Clean exit.")


if __name__ == "__main__":
    main()
