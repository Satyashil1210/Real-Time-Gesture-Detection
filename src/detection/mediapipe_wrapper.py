"""
MediaPipe Hands wrapper for RT-Gesture3D.

Responsibility:
    - Take a BGR frame (OpenCV)
    - Run MediaPipe Hands
    - Return per-hand landmarks in pixel coordinates
"""

from typing import List, Tuple
import cv2
import mediapipe as mp

Point3D = Tuple[int, int, float]  # (x, y, z)


class MediaPipeHandDetector:
    """
    Thin wrapper around MediaPipe Hands.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.6,   # 🔥 increased (better detection)
        tracking_confidence: float = 0.6,    # 🔥 smoother tracking
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils

        self._hands = self._mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def detect(self, frame_bgr) -> List[List[Point3D]]:
        """
        Returns:
            List of hands.
            Each hand = list of 21 (x, y, z) points in pixel coordinates.
        """

        if frame_bgr is None:
            return []

        h, w, _ = frame_bgr.shape

        # 🔥 PERFORMANCE BOOST: reduce resolution slightly
        small_frame = cv2.resize(frame_bgr, (640, 480))

        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        all_hands: List[List[Point3D]] = []

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                pts: List[Point3D] = []

                for lm in hand_lms.landmark:
                    # 🔥 scale back to original frame size
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    # 🔥 clamp values (avoid out-of-bound errors)
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))

                    pts.append((x, y, lm.z))

                all_hands.append(pts)

        return all_hands

    def draw_on_frame(self, frame_bgr) -> None:
        """
        Draw landmarks on frame (for debugging)
        """

        if frame_bgr is None:
            return

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_lms,
                    self._mp_hands.HAND_CONNECTIONS,
                )

    def close(self) -> None:
        self._hands.close()