import cv2
import mediapipe as mp
import math
from pathlib import Path
from ultralytics import YOLO


# ==============================
# 🧠 OBJECT DETECTOR (UPDATED)
# ==============================
class ObjectDetector:
    def __init__(self, model_path="yolov8m.pt", conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # 🔥 PRIORITY OBJECTS
        self.priority_classes = [
            "person", "chair", "laptop",
            "cell phone", "bottle", "book", "remote"
        ]

        # 🔥 EXTRA OBJECTS
        self.extra_classes = [
            "sofa", "bed", "keyboard", "mouse"
        ]

        # 🔥 CUSTOM LABELS
        self.custom_labels = {
            "airplane": "fan",
            "tv": "projector",
            "remote": "switch board",
            "backpack": "bag",
            "cell phone": "mobile",
        }

    def detect(self, frame):
        if frame is None:
            return []

        results = self.model(frame, verbose=False)

        priority_detections = []
        extra_detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                # 🔥 REMOVE NOISE
                if w * h < 2000:
                    continue

                # 🔥 FIX: chair → book
                if label == "chair":
                    aspect_ratio = w / (h + 1e-5)
                    if aspect_ratio < 0.75 and h > 120:
                        label = "book"

                # 🔥 CUSTOM LABEL
                if label in self.custom_labels:
                    label = self.custom_labels[label]

                obj = {
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                }

                # 🔥 PRIORITY / EXTRA SPLIT
                if label in self.priority_classes or label in self.custom_labels.values():
                    priority_detections.append(obj)
                elif label in self.extra_classes:
                    extra_detections.append(obj)

        # 🔥 SORT
        priority_detections = sorted(priority_detections, key=lambda x: x["confidence"], reverse=True)
        extra_detections = sorted(extra_detections, key=lambda x: x["confidence"], reverse=True)

        # 🔥 COMBINE
        return priority_detections[:3] + extra_detections[:2]


# ==============================
# 🎨 COLOR SYSTEM
# ==============================
def get_color(label):
    color_map = {
        "person": (0, 255, 0),
        "fan": (255, 0, 0),
        "book": (0, 255, 255),
        "mobile": (255, 255, 0),
        "laptop": (255, 0, 255),
        "bottle": (0, 128, 255),
        "chair": (0, 0, 255),
        "bag": (128, 0, 128),
    }
    return color_map.get(label, (200, 200, 200))


# ==============================
# Gesture helper functions (UNCHANGED)
# ==============================

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_extended_states(landmarks):
    tips_idx = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    pips_idx = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

    states = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}

    for name in ["index", "middle", "ring", "pinky"]:
        tip_y = landmarks[tips_idx[name]][1]
        pip_y = landmarks[pips_idx[name]][1]
        states[name] = (tip_y < pip_y - 2)

    wrist_x = landmarks[0][0]
    thumb_tip_x = landmarks[4][0]
    thumb_ip_x = landmarks[3][0]

    if abs(thumb_tip_x - wrist_x) > 30:
        states["thumb"] = True
    else:
        states["thumb"] = abs(thumb_tip_x - thumb_ip_x) > 15

    return states


def detect_gesture_from_landmarks(pts, img_w, img_h):
    st = finger_extended_states(pts)
    ext_count = sum(st.values())

    thumb_tip = (pts[4][0], pts[4][1])
    index_tip = (pts[8][0], pts[8][1])

    d_thumb_index = _dist(thumb_tip, index_tip)
    scale_thresh = max(40, int(img_w * 0.07))

    if d_thumb_index < scale_thresh and st["thumb"] and st["index"]:
        return 3, "perfect", 0.95

    if sum([st["index"], st["middle"], st["ring"], st["pinky"]]) >= 4:
        return 4, "stop", 0.9

    if st["index"] and st["pinky"] and not st["middle"] and not st["ring"]:
        return 5, "rock", 0.9

    if st["index"] and st["middle"]:
        if not st["ring"] and not st["pinky"]:
            return 1, "victory", 0.95
        if not st["ring"] or not st["pinky"]:
            return 1, "victory", 0.9

    if st["index"] and not st["middle"] and not st["ring"] and not st["pinky"]:
        return 6, "calm", 0.9

    if st["thumb"] and not any([st["index"], st["middle"], st["ring"], st["pinky"]]):
        return 2, "ok", 0.9

    return 0, "neutral", 0.5


# ==============================
# Avatar handling (UNCHANGED)
# ==============================

AVATAR_FILES = {
    "neutral": "neutral.jpg",
    "victory": "victory.jpg",
    "ok": "ok.jpg",
    "perfect": "perfect.jpg",
    "stop": "stop.jpg",
    "rock": "rock.jpg",
    "calm": "calm.jpg",
}


def load_avatars(size=(150, 150)):
    base_dir = Path(__file__).resolve().parents[2]
    assets_dir = base_dir / "assets" / "avtar"

    avatars = {}
    for label, fname in AVATAR_FILES.items():
        path = assets_dir / fname
        if not path.exists():
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        avatars[label] = cv2.resize(img, size)

    return avatars


def overlay_avatar(frame, avatar_img):
    if avatar_img is None:
        return frame

    fh, fw = frame.shape[:2]
    ah, aw = avatar_img.shape[:2]

    x1, y1 = fw - aw - 10, 10
    x2, y2 = x1 + aw, y1 + ah

    if x1 < 0 or y2 > fh:
        return frame

    frame[y1:y2, x1:x2] = avatar_img
    return frame


# ==============================
# 🚀 MAIN LOOP (UPDATED)
# ==============================

def main():
    print("▶️ Starting RT-Gesture3D demo...")

    avatars = load_avatars()
    obj_detector = ObjectDetector()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    frame_count = 0
    cached_objects = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # ======================
        # 📦 OBJECT DETECTION
        # ======================
        frame_count += 1
        if frame_count % 5 == 0:
            cached_objects = obj_detector.detect(frame)

        for obj in cached_objects:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label"]
            conf = obj["confidence"]

            color = get_color(label)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ======================
        # ✋ GESTURE DETECTION
        # ======================
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        label_text = "neutral"
        conf_text = 0.0

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                pts = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in handLms.landmark]

                _, label_text, conf_text = detect_gesture_from_landmarks(pts, w, h)

        cv2.putText(frame, f"{label_text} ({conf_text:.2f})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        frame = overlay_avatar(frame, avatars.get(label_text))

        cv2.imshow("RT-Gesture3D", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()