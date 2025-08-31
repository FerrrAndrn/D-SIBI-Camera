# D'SIBI Camera

![License](https://img.shields.io/badge/License-MIT-yellow.svg)

D'SIBI Camera is a real-time detection system for SIBI (Indonesian Sign Language) gestures using YOLO. The model is trained on 2600 images (100 per class for 26 letters A-Z) and achieves over 90% detection accuracy. The system allows both automatic video recording and manual snapshot capture for documentation and evaluation.

---

## 📂 Folder Structure
- **sibienv/** : Python environment (dependencies in `requirements.txt`)  
- **records/** : Automatic video recordings (`record_sibi1.mp4`, `record_sibi2.mp4`, ...)  
- **captures/** : Manually captured images (`<label>_<confidence_score>.png`)  
- **model/** : Trained YOLO model (`best.pt`)  
- **realtimedetection.py** : Main script for real-time detection  

---

## ⚙️ Features

### 1. Save Detection Image (Key **S**)
- Press **S** to save the current detection frame manually.  
- Images are saved in `captures/` with filename format: `<label>_<confidence_score>.png`  
- Example: `L_0.94.png` → gesture "L" detected with 0.94 confidence  

### 2. Exit Program (Key **Q**)
- Press **Q** to stop detection immediately.  
- Closes camera window and releases video capture safely.  

### 3. Automatic Video Recording
- Recording starts automatically when the camera is activated.  
- Videos are saved in `records/` with sequential filenames:  
  `record_sibi1.mp4`, `record_sibi2.mp4`, ...  
- Each frame includes bounding boxes, gesture labels, confidence scores, and FPS.  
- Useful for documentation and performance evaluation.  

---

## 🚀 How to Run
```bash
# Install dependencies
pip install -r sibienv/requirements.txt
o Use S to save snapshots, Q to exit.

📝 Notes

Requires access to a camera.

GPU with CUDA recommended for optimal performance.

Output files are organized in captures/ and records/ directories.

📜 License & HKI

This project is registered under Indonesian HKI (Hak Kekayaan Intelektual).
Unauthorized commercial use, reproduction, or distribution is strictly prohibited.
Contact: [Your Email] for permission or licensing inquiries.

# Run real-time detection
python realtimedetection.py --weights model/best.pt
