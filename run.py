import sys
import os

# 🔥 project root path add karo
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 👉 main function import karo
from src.inference.live_gesture_main import main


if __name__ == "__main__":
    print("🚀 Running RT-Gesture3D from run.py")
    main()