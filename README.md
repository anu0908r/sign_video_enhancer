# 🎬 Video Enhancer

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Ready-yellow?style=for-the-badge&logo=google)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)

Video Enhancer is a powerful, Python-based toolset designed to stabilize shaky videos, enhance facial and eye clarity, and seamlessly convert video formats. Built with advanced computer vision libraries, it delivers high-quality video processing for creators and developers alike.

## ✨ Key Features

*   **🎥 Video Stabilization:** Remove jitter and camera shake from your videos. Features both custom optical flow stabilization and integration with the robust `vidstab` library.
*   **👁️ Face & Eye Clarity Enhancement:** Automatically detect faces and eyes in video frames and apply localized sharpening for improved clarity. Supports both OpenCV cascades and MediaPipe Face Mesh.
*   **🎯 Face-Lock Stabilization:** Center the video on detected faces to keep the subject locked in the frame automatically.
*   **🔄 Format Conversion:** Easily convert stabilized output videos to H.264 format for broad compatibility across devices and web players.
*   **🖼️ Frame Extraction:** Extract individual frames from videos for frame-by-frame analysis, processing, and debugging.

## 🛠️ Tech Stack

**Core Logic & Processing:**
*   Python 3
*   OpenCV (Computer Vision)
*   NumPy (Matrix Operations)

**AI & Advanced Features:**
*   MediaPipe (Advanced Face Mesh & Landmarks)
*   VidStab (Frame-by-frame Stabilization)
*   ImageIO (Video Format Conversion)

## 🚀 Getting Started

### Prerequisites
Before running this project, ensure you have the following installed:
*   [Python](https://www.python.org/downloads/) (v3.8 or higher recommended)
*   pip (Python package manager)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/video-enhancer.git
    cd video-enhancer
    ```

2.  **Install dependencies:**
    ```bash
    pip install opencv-python numpy mediapipe vidstab imageio
    ```
    *(Note: Depending on your environment, you may also need `imageio[ffmpeg]` for video conversion capabilities).*

3.  **Prepare your directories:**
    Ensure you have an input directory with your source videos (the default is `input1/`):
    ```bash
    mkdir input1
    # Place your input video in the input1 directory
    ```

4.  **Run the core enhancement script:**
    ```bash
    python main.py --input input1/my_video.mp4 --output output1/frames --video-output output1/final_video.mp4
    ```

## 📂 Project Structure

```text
video_enhancer/
├── main.py                # Core pipeline for extraction, stabilization & clarity
├── vidstab_stabilize.py   # Dedicated fast stabilization using the vidstab library
├── convert.py             # Utility to convert outputs to H.264 format
├── input1/                # Default directory for input source videos
└── output1/               # Default directory for processed frames and output videos
