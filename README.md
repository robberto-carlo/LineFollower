# LineFollower

<p align="center">
  <img src="https://img.shields.io/badge/Device-Raspberry%20Pi%204-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Vision%20System-OpenCV-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Version-v1.0-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge">
</p>

---

##  Overview

This repository documents the development of my personal project: a **camera-based line-following robot**.

The system is designed to run on a **Raspberry Pi 4** with a camera, without using infrared sensors. Instead, it relies entirely on **real-time computer vision processing** to detect the line and make movement decisions.

Throughout the project, I have implemented multiple OpenCV-based vision modules, including line detection, position analysis, and the recognition of special cases such as intersections, curves, and line endings.

---

##  System Logic

The robot processes each frame captured by the camera using OpenCV. The image is converted to grayscale and a threshold is applied to isolate the line from the background. Then, the system analyzes the shape and position of the detected region to determine the robot’s motion.

### Detected states:

- Normal line following
- Curves using centroid deviation analysis
- “T” intersections using green color detection for decision-making
- End of line detection
- Loss of signal or unclear frames

---

## Color Calibration System

This project includes a real-time calibration tool that allows tuning segmentation values using interactive trackbars.

### Features:

- Adjust **minimum and maximum threshold values**
- Real-time camera visualization
- Calibration for line and color detection
- Save optimal parameters for competition use

This allows the system to adapt to different lighting conditions without modifying the main code.

---
## Requirements

### Hardware

- Raspberry Pi 4 (Embedded system)
- Camera module


### Dependencies

- Python 3.9
- OpenCV
- NumPy

---

## 🎯 Objective

The main objective is to build a fully autonomous navigation system based purely on computer vision, capable of interpreting its environment and making real-time decisions in a reliable, stable, and adaptable way.