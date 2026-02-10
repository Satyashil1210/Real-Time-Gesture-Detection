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

    Example:
        detector = MediaPipeHandDetector()
        hands_pts = detector.detect(frame)
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
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
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        all_hands: List[List[Point3D]] = []
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                pts: List[Point3D] = []
                for lm in hand_lms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    pts.append((x, y, lm.z))
                all_hands.append(pts)

        return all_hands

    def draw_on_frame(self, frame_bgr) -> None:
        """
        Convenience: re-run mediapipe and draw landmarks on the given frame.
        (Not used in main pipeline yet, but handy for quick debugging.)
        """
        h, w, _ = frame_bgr.shape
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
