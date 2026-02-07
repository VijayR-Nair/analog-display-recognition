# Analog Display Recognition System

Computer vision and deep learning system to convert real-time analog meter readings into digital values using ESP32-CAM, TensorFlow, and OpenCV.

---

## Overview
This project automates the reading of analog meters by detecting the needle, zero markings, and dial center from live camera frames and computing accurate measurements using a geometry-based algorithm.

---

## Tech Stack
- Python
- TensorFlow (Faster R-CNN)
- OpenCV
- Tkinter (GUI)
- ESP32-CAM

---

![Mount Design](3D_Model.png)

## How it Works
1. ESP32-CAM streams live images
2. Deep learning model detects meter components
3. Geometry algorithm computes the value
4. Result is displayed on a GUI in real time

---

## Results
- Accuracy: >97%
- Works in real time
- Robust to fast needle movement

---

## Notes
- Trained model and dataset are not included
- Update model paths in `main_adrs.py` before running
