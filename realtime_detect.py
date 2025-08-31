from ultralytics import YOLO
import cv2
import time
import os

def get_next_video_filename(prefix="record_sibi", ext="mp4", folder="."):
    i = 1
    while True:
        filename = os.path.join(folder, f"{prefix}{i}.{ext}")
        if not os.path.exists(filename):
            return filename
        i += 1

def get_unique_filename(folder, base_name, ext="png"):
    filename = f"{base_name}.{ext}"
    filepath = os.path.join(folder, filename)
    i = 1
    while os.path.exists(filepath):
        filename = f"{base_name}_{i}.{ext}"
        filepath = os.path.join(folder, filename)
        i += 1
    return filepath

model = YOLO("best.pt")

save_dir = "captures"
video_dir = "records"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1)

if not cap.isOpened():
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera_fps = 8

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_filename = get_next_video_filename(prefix="record_sibi", ext="mp4", folder=video_dir)
video_writer = cv2.VideoWriter(video_filename, fourcc, camera_fps, (width, height))

box_color = (128, 0, 0)
fps_color = (0, 255, 255)
frame_count = 0
detection_interval = 1
prev_time = time.time()
last_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % detection_interval == 0:
        results = model.predict(frame, conf=0.5, verbose=False, device='cpu')
        last_detections.clear()
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                conf = float(box.conf[0]) * 100
                last_detections.append((x1, y1, x2, y2, label, conf))

    for (x1, y1, x2, y2, label, conf) in last_detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f'{label} {conf:.1f}%', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-5)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, fps_color, 2)

    cv2.imshow("SKRIPSI SIBI Detection", frame)
    video_writer.write(frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s") and last_detections:
        label = last_detections[0][4]
        conf = last_detections[0][5]
        base_filename = f"{label}_{conf:.1f}"
        unique_filepath = get_unique_filename(save_dir, base_filename)
        cv2.imwrite(unique_filepath, frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
