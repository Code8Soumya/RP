import streamlit as st
import cv2
import time
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage
from ultralytics import YOLO
import threading

# Configuration
st.set_page_config(page_title="3D Print Monitor", layout="wide")
st.title("🖨️ 3D Printing Defect Monitor")

# --- Sidebar for Settings ---
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", min_value=0.1, max_value=0.95, value=0.50, step=0.05)
ip_camera_url = st.sidebar.text_input("IP Camera URL (e.g., http://192.168.1.xxx:8080/video)", "")
st.sidebar.markdown("### Email Alert Setup")
st.sidebar.caption("For Gmail, use an **App Password** instead of your regular password.")
sender_email = st.sidebar.text_input("Sender Email")
sender_password = st.sidebar.text_input("Sender App Password", type="password")
receiver_email = st.sidebar.text_input("Receiver Email")
cooldown_period = st.sidebar.number_input("Alert Cooldown (seconds)", min_value=10, value=60)

# Ensure defects folder exists
if not os.path.exists("defects"):
    os.makedirs("defects")

# Load YOLO Model
@st.cache_resource
def load_model():
    model_path = "models/best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error(f"⚠️ Model not found at `{model_path}`! Please make sure your model is placed there.")
        return None

model = load_model()

def send_email_async(subject, body, image_path, sender, password, receiver):
    """Sends email in a background thread to prevent pausing the video stream."""
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
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")

    threading.Thread(target=send).start()

start_button = st.sidebar.button("Start Monitoring")
stop_button = st.sidebar.button("Stop Monitoring")

if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False

# Manage Start/Stop state
if start_button:
    st.session_state.monitoring = True
if stop_button:
    st.session_state.monitoring = False

# Flashing CSS
st.markdown("""
<style>
.blinking {
    animation: blinker 1s linear infinite;
    color: white;
    background-color: #ff4b4b;
    font-weight: bold;
    font-size: 20px;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 10px;
    border: 2px solid darkred;
}
@keyframes blinker {
    50% { opacity: 0; }
}
</style>
""", unsafe_allow_html=True)

if 'defect_history' not in st.session_state:
    st.session_state.defect_history = []

# Split layout into two columns: Live Video (70%) and Defect Photos (30%)
col1, col2 = st.columns([7, 3], gap="large")

with col1:
    st.markdown("### 🎥 Live Video Feed")
    alert_placeholder = st.empty()
    frame_placeholder = st.empty()

with col2:
    st.markdown("### 🛑 Defect History")
    history_placeholder = st.empty()

def update_history_ui():
    """Helper function to draw the defect history in the right column"""
    with history_placeholder.container():
        if not st.session_state.defect_history:
            st.info("No defects detected yet.")
        for item in st.session_state.defect_history:
            st.image(item["image"], caption=f"{item['time']} - {item['defect']}", width="stretch")
            st.markdown("---")

# Initial draw of history
update_history_ui()

if st.session_state.monitoring:
    if not ip_camera_url or not model:
        st.warning("Please provide a valid IP Camera URL and ensure your YOLO model exists.")
    else:
        # Capture Video Stream
        cap = cv2.VideoCapture(ip_camera_url)
        last_alert_time = 0
        current_defect_name = ""

        while cap.isOpened() and st.session_state.monitoring:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from camera. Reconnecting...")
                time.sleep(2)
                cap = cv2.VideoCapture(ip_camera_url)
                continue

            # Resize frame to avoid running out of RAM (Out of Memory Error)
            frame = cv2.resize(frame, (640, 480))
            
            # Run YOLO inference explicitly specifying image size to save memory
            results = model(frame, verbose=False, imgsz=640, conf=confidence_threshold)
            annotated_frame = results[0].plot()
            
            # Convert BGR to RGB right away since we might use it for saving history
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Parse bounding boxes for detections
            boxes = results[0].boxes
            current_time = time.time()
            
            if len(boxes) > 0:
                # Highest confidence defect
                best_conf_index = boxes.conf.argmax()
                defect_class_id = int(boxes.cls[best_conf_index])
                current_defect_name = model.names[defect_class_id]
                
                # Check cooldown to avoid spamming alerts & emails
                if current_time - last_alert_time > cooldown_period:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    human_readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Save image locally
                    image_filename = f"defects/{current_defect_name}_{timestamp}.jpg"
                    cv2.imwrite(image_filename, annotated_frame)
                    
                    # Update History Session State
                    st.session_state.defect_history.insert(0, {
                        "time": human_readable_time,
                        "defect": current_defect_name.upper(),
                        "image": frame_rgb
                    })
                    # Keep latest 10 to prevent UI lag/memory issues
                    if len(st.session_state.defect_history) > 10:
                        st.session_state.defect_history.pop()
                        
                    # Re-draw the history panel
                    update_history_ui()
                    
                    # Send Email
                    if sender_email and sender_password and receiver_email:
                        email_body = f"Hello,\n\nA defect classified as '{current_defect_name}' was detected on your 3D printer at {human_readable_time}.\n\nPlease check the attached image."
                        send_email_async(
                            subject="⚠️ 3D Print Defect Alert",
                            body=email_body,
                            image_path=image_filename,
                            sender=sender_email,
                            password=sender_password,
                            receiver=receiver_email
                        )
                    
                    last_alert_time = current_time

            # Flashing UI logic (Flash for 5 seconds after a detection)
            if current_time - last_alert_time < 5:
                alert_html = f'<div class="blinking">🚨 DEFECT DETECTED: {current_defect_name.upper()} 🚨</div>'
                alert_placeholder.markdown(alert_html, unsafe_allow_html=True)
            else:
                alert_placeholder.empty()

            # Render the live video frame
            frame_placeholder.image(frame_rgb, channels="RGB", width="stretch")

            # Prevent high CPU usage in endless loops
            time.sleep(0.01)

        cap.release()
else:
    st.info("Monitoring is currently stopped. Enter your settings and click 'Start Monitoring'.")





