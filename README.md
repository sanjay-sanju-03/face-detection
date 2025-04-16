# Stall Entry Face Recognition System

A Python-based real-time face recognition system for monitoring and controlling entry into stalls or booths. Prevents duplicate entries by detecting previously registered faces and playing an alert sound if the same person attempts to enter again.

## 🚀 Features

- Real-time face detection using Mediapipe.
- Face recognition with OpenCV's LBPH algorithm.
- Duplicate face alert system with audio notification.
- Auto-training after a defined number of new faces.
- Simple and intuitive UI using OpenCV window.

## 🛠️ Installation

``bash
# Clone the repository
git clone https://github.com/sanjay-sanju-03/face-detection.git

# Navigate into the project folder
cd face-detection

# Install Python dependencies
pip install opencv-python mediapipe playsound numpy

To run the file
python main.py
