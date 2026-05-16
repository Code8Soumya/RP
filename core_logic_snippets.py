# --- 1. Load YOLO Model ---
from ultralytics import YOLO
import os

def load_model(model_path="models/best.pt"):
    """Loads the YOLO model from the specified path."""
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

# --- 2. Asynchronous Email Alert System ---
import smtplib
from email.message import EmailMessage
import threading

def send_email_async(subject, body, image_path, sender, password, receiver):
    """Sends email in a background thread to prevent pausing the video stream during monitoring."""
    def send():
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = receiver
        msg.set_content(body)

        with open(image_path, 'rb') as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender, password)
                server.send_message(msg)
        except Exception as e:
            print(f"Failed to send email: {e}")

    threading.Thread(target=send).start()

# --- 3. Core Video Processing & Inference Loop ---
import cv2
import time
from datetime import datetime

def monitor_stream(ip_camera_url, model, confidence_threshold, cooldown_period):
    """Core logic for reading IP camera stream and running YOLO inference."""
    cap = cv2.VideoCapture(ip_camera_url)
    last_alert_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to save memory
        frame = cv2.resize(frame, (640, 480))
        
        # YOLO inference
        results = model(frame, verbose=False, imgsz=640, conf=confidence_threshold)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes
        current_time = time.time()
        
        if len(boxes) > 0:
            # Get highest confidence defect
            best_conf_index = boxes.conf.argmax()
            defect_class_id = int(boxes.cls[best_conf_index])
            current_defect_name = model.names[defect_class_id]
            
            # Cooldown logic to prevent spam
            if current_time - last_alert_time > cooldown_period:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                image_filename = f"defects/{current_defect_name}_{timestamp}.jpg"
                
                # Save annotated image for history/emails
                cv2.imwrite(image_filename, annotated_frame)
                
                # (Call send_email_async here)
                
                last_alert_time = current_time

    cap.release()




