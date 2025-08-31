D'SIBI Camera

Description:
D'SIBI Camera is a real-time detection system for SIBI (Indonesian Sign Language) gestures using YOLO. Trained on 2600 images (100 per class for 26 letters A-Z), the model achieves over 90% accuracy. The system allows both automatic video recording and manual snapshot capture for documentation and evaluation.

Folder Structure:
- sibienv/           : Python environment (dependencies listed in requirements.txt)
- records/           : Stores automatic video recordings (record_sibi1.mp4, record_sibi2.mp4, ...)
- captures/          : Stores manually captured images (format: <label>_<confidence_score>.png)
- model/             : Contains trained YOLO model (best.pt)
- realtimedetection.py : Main script for real-time detection

Features:
1. Save Detection Image (Key S)
   - Press 'S' to save the current detection frame manually.
   - Image is saved in captures/ with filename format <label>_<confidence_score>.png
   - Example: L_0.94.png → gesture "L" detected with 0.94 confidence

2. Exit Program (Key Q)
   - Press 'Q' to immediately stop detection.
   - Closes camera window and releases video capture safely.

3. Automatic Video Recording
   - Starts automatically when the camera is activated.
   - Saves video in records/ with sequential filenames: record_sibi1.mp4, record_sibi2.mp4, ...
   - Each frame includes bounding boxes, gesture labels, confidence scores, and FPS.
   - Useful for documentation and performance evaluation.

How to Run:
1. Make sure Python dependencies are installed:
   pip install -r sibienv/requirements.txt
2. Run the detection script:
   python realtimedetection.py --weights model/best.pt
3. Use 'S' to save snapshots, 'Q' to exit.

Notes:
- The system requires access to a camera.
- GPU with CUDA is recommended for optimal performance.
- Output files are organized in captures/ and records/ directories.

License:
- This project is registered under Indonesian HKI (Hak Kekayaan Intelektual).
- Unauthorized use, reproduction, or distribution is prohibited.
- Contact [Your Email] for permission or licensing inquiries.
