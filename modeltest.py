from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob
import time
import torch

DETECTOR_PATH = 'GTSDB.pt'
CLASSIFIER_PATH = 'GTSRB.pt'
SOURCE_PATH = 'traffic-sign-test.mp4' 

RECORD_DIR = 'RunsRecordings'
os.makedirs(RECORD_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(RECORD_DIR, f"recorded_run_{int(time.time())}.mp4")


Detector_CONFIDENCE = 0.6 
Classifier_CONFIDENCE = 0.8 
Detector_Image_Size = 20*32   # 640
Classifier_Image_Size = 10*32 # 320


MIN_DETECTIONS = 2

RESIZE_W = 0.7 
RESIZE_H = 0.7 

ROI_Y_START = 0.15  
ROI_Y_END = 0.85    

ROI_X_START = 0  
ROI_X_END = 1    

MIN_SIGN_SIZE = 30  
MAX_SIGN_SIZE = 300 


TARGET_FPS = 30
FRAME_DELAY = 1.0 / TARGET_FPS

CLASS_NAMES = [
    '20_speed', '30_speed', '50_speed', '60_speed', '70_speed', '80_speed', '80_lifted', 
    '100_speed', '120_speed', 'no_overtaking_general', 'no_overtaking_trucks', 
    'right_of_way_crossing', 'right_of_way_general', 'give_way', 'stop', 
    'no_way_general', 'no_way_trucks', 'no_way_one_way', 'attention_general', 
    'attention_left_turn', 'attention_right_turn', 'attention_curvy', 'attention_bumpers', 
    'attention_slippery', 'attention_bottleneck', 'attention_construction', 
    'attention_traffic_light', 'attention_pedestrian', 'attention_children', 
    'attention_bikes', 'attention_snowflake', 'attention_deer', 'lifted_general', 
    'turn_right', 'turn_left', 'turn_straight', 'turn_straight_right', 'turn_straight_left', 
    'turn_right_down', 'turn_left_down', 'turn_circle', 'lifted_no_overtaking_general', 
    'lifted_no_overtaking_trucks'
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = YOLO(DETECTOR_PATH).to(device)
classifier = YOLO(CLASSIFIER_PATH).to(device)

last_speed_limit = "No Limit"
last_speed_crop = None 
side_panel_data = []
detection_buffer = [] 

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0: return 0
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def process_frame(frame):
    global last_speed_limit, last_speed_crop, detection_buffer, side_panel_data
    
    t_start = time.time()
    h_orig, w_orig = frame.shape[:2]
    frame = cv2.resize(frame, (int(w_orig * RESIZE_W), int(h_orig * RESIZE_H)))
    h, w = frame.shape[:2]
    display_frame = frame.copy()
    
    roi_y1, roi_y2 = int(h * ROI_Y_START), int(h * ROI_Y_END)
    roi_x1, roi_x2 = int(w * ROI_X_START), int(w * ROI_X_END)
    cv2.rectangle(display_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 255), 2)
    cv2.putText(display_frame, "ROI", (15, roi_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    results = detector.predict(frame, imgsz=Detector_Image_Size, conf=Detector_CONFIDENCE, verbose=False, device=device)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    det_scores = results[0].boxes.conf.cpu().numpy()

    updated_indices = set()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        det_conf = float(det_scores[i])

        center_y = (y1 + y2) // 2
        center_x = (x1 + x2) // 2
        if not (roi_y1 < center_y < roi_y2): continue
        if not (roi_x1 < center_x < roi_x2): continue
        bw, bh = x2 - x1, y2 - y1
        if not (MIN_SIGN_SIZE < bw < MAX_SIGN_SIZE and MIN_SIGN_SIZE < bh < MAX_SIGN_SIZE): continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue

        class_results = classifier.predict(crop, imgsz=Classifier_Image_Size, conf=Classifier_CONFIDENCE, verbose=False, device=device)
        
        label = "Unknown Sign"
        cls_conf = 0.0
        color = (0, 165, 255)
        
        if len(class_results[0].boxes) > 0:
            cls_id = int(class_results[0].boxes.cls[0])
            cls_conf = float(class_results[0].boxes.conf[0])
            label = CLASS_NAMES[cls_id]
            color = (0, 255, 0)
            
            print(f"[DETECTED] Name: {label} | Dim: {bw}x{bh} | D-Conf: {det_conf:.1%} | C-Conf: {cls_conf:.1%}")

            found_in_buffer = False
            current_count = 1
            
            for j, (b_box, b_label, b_count) in enumerate(detection_buffer):
                if get_iou(box, b_box) > 0.3 and label == b_label:
                    detection_buffer[j] = (box, label, b_count + 1)
                    updated_indices.add(j)
                    found_in_buffer = True
                    current_count = b_count + 1
                    break
            
            if not found_in_buffer:
                detection_buffer.append((box, label, 1))
                updated_indices.add(len(detection_buffer) - 1)
                current_count = 1

            if current_count >= MIN_DETECTIONS:
                res_crop = cv2.resize(crop, (120, 120))
                
                existing_idx = -1
                for idx, data in enumerate(side_panel_data):
                    if data[0] == label:
                        existing_idx = idx
                        break
                
                if existing_idx != -1:
                    side_panel_data[existing_idx] = [label, res_crop, current_count, det_conf, cls_conf]
                    item = side_panel_data.pop(existing_idx)
                    side_panel_data.insert(0, item)
                else:
                    side_panel_data.insert(0, [label, res_crop, current_count, det_conf, cls_conf])
                
                if len(side_panel_data) > 5: side_panel_data = side_panel_data[:5]

                if 'speed' in label or 'lifted' in label:
                    if 'speed' in label:
                        last_speed_limit = f"SPEED LIMIT: {label.split('_')[0]}"
                    else:
                        last_speed_limit = "LIMIT ENDED"
                    last_speed_crop = cv2.resize(crop, (50, 50))

        display_label = f"{label} D:{det_conf:.0%} C:{cls_conf:.0%}"
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, display_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


    detection_buffer = [item for idx, item in enumerate(detection_buffer) if idx in updated_indices]
    

    panel_w = 250
    ui_panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    for i, (name, img, count, d_conf, c_conf) in enumerate(side_panel_data):
        y_pos = 30 + (i * 170)
        if y_pos + 120 < h:
            ui_panel[y_pos:y_pos+120, 65:185] = img
            cv2.putText(ui_panel, f"{name[:12]} (x{count})", (15, y_pos+140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(ui_panel, f"D:{d_conf:.1%} C:{c_conf:.1%}", (15, y_pos+158), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    final_output = np.hstack((display_frame, ui_panel))
    
    overlay = final_output.copy()
    cv2.rectangle(overlay, (w//2 - 180, 5), (w//2 + 180, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, final_output, 0.6, 0, final_output)
    
    status_color = (0, 0, 255) if "LIMIT" in last_speed_limit and "ENDED" not in last_speed_limit else (0, 255, 0)
    cv2.putText(final_output, last_speed_limit, (w//2 - 100, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    if last_speed_crop is not None:
        final_output[8:58, w//2 - 170 : w//2 - 120] = last_speed_crop

    proc_duration = time.time() - t_start
    fps = 1.0 / proc_duration if proc_duration > 0 else TARGET_FPS
    cv2.putText(final_output, f"FPS: {fps:.1f}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return final_output, proc_duration

cv2.namedWindow("Autonomous Sign System", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(SOURCE_PATH)

out_video = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    out, duration = process_frame(frame)
    cv2.imshow("Autonomous Sign System", out)
    
    if out_video is None:
        height, width, _ = out.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(OUTPUT_PATH, fourcc, TARGET_FPS, (width, height))
        print(f"[INFO] Recording started. Output will be saved to: {OUTPUT_PATH}")
    
    out_video.write(out)
    
    wait_time = max(1, int((FRAME_DELAY - duration) * 1000))
    if cv2.waitKey(wait_time) & 0xFF == ord('q'): break

cap.release()
if out_video is not None:
    out_video.release()
cv2.destroyAllWindows()
print("[INFO] Recording finished and saved successfully.")
