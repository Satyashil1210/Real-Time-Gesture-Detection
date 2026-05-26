"""
object_detector.py — RT-Gesture3D v2
====================================

Optimized YOLOv8 Object Detector.

Features
--------
✅ Fast YOLO inference
✅ Smart filtering
✅ Priority objects
✅ Custom real-world labels
✅ FPS monitoring
✅ Noise reduction
✅ Meaningful output support
"""

from __future__ import annotations

import time

import cv2
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# Object Detector
# ─────────────────────────────────────────────────────────────

class ObjectDetector:

    def __init__(
        self,
        model_path="yolov8m.pt",
        conf_threshold=0.4,
        debug=False,
    ):

        # YOLO
        self.model = YOLO(model_path)

        # settings
        self.conf_threshold = conf_threshold

        self.debug = debug

        # ─────────────────────────────────────────────────────
        # Priority Objects
        # ─────────────────────────────────────────────────────

        self.priority_classes = [

            "person",

            "backpack",

            "chair",

            "laptop",

            "cell phone",

            "bottle",

            "tv",

            "book",

            "cup",

            "remote",
        ]

        # ─────────────────────────────────────────────────────
        # Secondary Objects
        # ─────────────────────────────────────────────────────

        self.extra_classes = [

            "sofa",

            "bed",

            "potted plant",

            "keyboard",

            "mouse",

            "toilet",
        ]

        # ─────────────────────────────────────────────────────
        # Meaningful Custom Labels
        # ─────────────────────────────────────────────────────

        self.custom_labels = {

            "airplane": "fan",

            "tv": "projector",

            "remote": "switch board",

            "backpack": "bag",

            "cell phone": "mobile",

            "dining table": "table",
        }

        print("✅ YOLO Object Detector Ready")

    # ─────────────────────────────────────────────────────────
    # Detect
    # ─────────────────────────────────────────────────────────

    def detect(self, frame):

        start_time = time.time()

        if frame is None:
            return []

        # ─────────────────────────────────────────────────────
        # RGB Conversion
        # ─────────────────────────────────────────────────────

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ─────────────────────────────────────────────────────
        # YOLO Inference
        # ─────────────────────────────────────────────────────

        results = self.model.predict(

            rgb,

            imgsz=640,

            conf=0.25,

            verbose=False,
        )

        priority_detections = []

        extra_detections = []

        # ─────────────────────────────────────────────────────
        # Parse Results
        # ─────────────────────────────────────────────────────

        for result in results:

            if result.boxes is None:
                continue

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            scores = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
            )

            for box, score, cls in zip(
                boxes,
                scores,
                classes
            ):

                # confidence
                if score < self.conf_threshold:
                    continue

                label = self.model.names[
                    int(cls)
                ]

                # bbox
                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                width = x2 - x1

                height = y2 - y1

                area = width * height

                # ─────────────────────────────────────────────
                # Remove Small Noise
                # ─────────────────────────────────────────────

                if area < 2000:
                    continue

                # ─────────────────────────────────────────────
                # Smart Chair→Book Fix
                # ─────────────────────────────────────────────

                if label == "chair":

                    aspect = width / (height + 1e-5)

                    if aspect < 0.75 and height > 120:

                        label = "book"

                # ─────────────────────────────────────────────
                # Meaningful Mapping
                # ─────────────────────────────────────────────

                if label in self.custom_labels:

                    label = self.custom_labels[
                        label
                    ]

                # ─────────────────────────────────────────────
                # Object Dict
                # ─────────────────────────────────────────────

                obj = {

                    "label": label,

                    "confidence": float(score),

                    "bbox": [

                        x1,
                        y1,
                        x2,
                        y2
                    ]
                }

                # ─────────────────────────────────────────────
                # Priority
                # ─────────────────────────────────────────────

                if (

                    label in self.priority_classes

                    or

                    label in self.custom_labels.values()
                ):

                    priority_detections.append(
                        obj
                    )

                elif label in self.extra_classes:

                    extra_detections.append(
                        obj
                    )

        # ─────────────────────────────────────────────────────
        # Sorting
        # ─────────────────────────────────────────────────────

        priority_detections = sorted(

            priority_detections,

            key=lambda x: x["confidence"],

            reverse=True
        )

        extra_detections = sorted(

            extra_detections,

            key=lambda x: x["confidence"],

            reverse=True
        )

        # ─────────────────────────────────────────────────────
        # Final Objects
        # ─────────────────────────────────────────────────────

        final_detections = (

            priority_detections[:3]

            +

            extra_detections[:2]
        )

        # ─────────────────────────────────────────────────────
        # FPS Debug
        # ─────────────────────────────────────────────────────

        if self.debug:

            end_time = time.time()

            inference_time = (
                end_time - start_time
            ) * 1000

            fps = (
                1000 / inference_time
                if inference_time > 0
                else 0
            )

            print(
                f"⏱ Inference:"
                f" {inference_time:.2f} ms"
            )

            print(
                f"🚀 FPS:"
                f" {fps:.2f}"
            )

            print(
                "🔥 Objects:",
                final_detections
            )

        return final_detections