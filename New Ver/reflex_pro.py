import cv2
import mediapipe as mp
import time
import math
import threading
import subprocess
import atexit
import json
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

# ================= CONFIGURATION =================
STREAM_URL = "tcp://127.0.0.1:8888"
EYE_AR_THRESH = 0.22    # Blink Threshold
SHAKE_THRESH = 10.0     # Head Movement Sensitivity
GAZE_THRESH = 0.3       # Gaze Sensitivity (Higher = look further to trigger)

# ================= FLASK & STATE =================
app = Flask(__name__)

# Global Status for Chrome (Thread Safe)
current_status = {
    "alert": None,      # 'blink', 'shake', 'left', 'right'
    "face_detected": False
}

# ================= CAMERA HARDWARE =================
camera_process = subprocess.Popen([
    "rpicam-vid", "-t", "0", "--width", "640", "--height", "480",
    "--framerate", "30", "--codec", "mjpeg", "--inline", "--listen", "-o", STREAM_URL
])
atexit.register(camera_process.terminate)
time.sleep(2.0)

# ================= THREADED STREAM =================
class CameraStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(STREAM_URL)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                self.grabbed = True
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

vs = CameraStream().start()

# ================= AI & LOGIC =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # NEEDED for Iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNERS = [33, 133] # Outer, Inner
RIGHT_EYE_CORNERS = [362, 263] # Inner, Outer

def get_gaze_ratio(landmarks, iris_idx, corners_idx, width):
    # Get Iris Center
    iris_pts = np.array([(landmarks[i].x * width, landmarks[i].y) for i in iris_idx])
    iris_center = np.mean(iris_pts, axis=0)
    
    # Get Eye Corners
    left_corner = np.array([landmarks[corners_idx[0]].x * width, landmarks[corners_idx[0]].y])
    right_corner = np.array([landmarks[corners_idx[1]].x * width, landmarks[corners_idx[1]].y])
    
    # Calculate relative position (0.0 = Left, 0.5 = Center, 1.0 = Right)
    eye_width = np.linalg.norm(right_corner - left_corner)
    dist_to_left = np.linalg.norm(iris_center - left_corner)
    
    ratio = dist_to_left / eye_width
    return ratio

def calculate_ear(landmarks, width, height):
    # Simple Blink Logic
    # Left Eye (33, 160, 158, 133, 153, 144)
    # Vertical distance
    v1 = abs(landmarks[160].y - landmarks[144].y)
    v2 = abs(landmarks[158].y - landmarks[153].y)
    # Horizontal
    h = abs(landmarks[33].x - landmarks[133].x)
    return (v1 + v2) / (2.0 * h)

def generate_frames():
    prev_nose = None
    
    while True:
        frame = vs.read()
        if frame is None: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        alert_type = None
        
        if results.multi_face_landmarks:
            current_status["face_detected"] = True
            lm = results.multi_face_landmarks[0].landmark
            
            # --- 1. FACE BOUNDING BOX ---
            x_vals = [l.x * w for l in lm]
            y_vals = [l.y * h for l in lm]
            x_min, x_max = int(min(x_vals)), int(max(x_vals))
            y_min, y_max = int(min(y_vals)), int(max(y_vals))
            
            # Draw Clean Box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (200, 200, 200), 1)

            # --- 2. SHAKE DETECTION ---
            nose_x, nose_y = lm[1].x * w, lm[1].y * h
            if prev_nose is not None:
                movement = math.hypot(nose_x - prev_nose[0], nose_y - prev_nose[1])
                if movement > SHAKE_THRESH:
                    alert_type = "shake"
            prev_nose = (nose_x, nose_y)

            # --- 3. BLINK DETECTION ---
            ear = calculate_ear(lm, w, h)
            if ear < EYE_AR_THRESH:
                alert_type = "blink"

            # --- 4. GAZE DETECTION (Right/Left) ---
            # Ratio: < 0.4 is Looking Right (Mirror), > 0.6 is Looking Left
            # Note: Because of Mirror Flip, logic reverses
            r_ratio = get_gaze_ratio(lm, RIGHT_IRIS, RIGHT_EYE_CORNERS, w)
            l_ratio = get_gaze_ratio(lm, LEFT_IRIS, LEFT_EYE_CORNERS, w)
            avg_gaze = (r_ratio + l_ratio) / 2

            if avg_gaze < (0.5 - 0.05): # Tuned for stability
                alert_type = "right"
            elif avg_gaze > (0.5 + 0.05):
                alert_type = "left"

            # --- VISUALS ---
            color = (0, 255, 0)
            text = "CLEAN"
            
            if alert_type == "shake":
                color = (0, 0, 255)
                text = "SHAKE DETECTED"
            elif alert_type == "blink":
                color = (0, 255, 255)
                text = "BLINK DETECTED"
            elif alert_type == "left":
                color = (255, 0, 255)
                text = "LOOKING LEFT"
            elif alert_type == "right":
                color = (255, 0, 255)
                text = "LOOKING RIGHT"

            current_status["alert"] = alert_type
            
            # Corner Alerts
            if alert_type:
                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Eye Landmarks (Thin)
            for idx in [33, 133, 362, 263, 468, 473]:
                lx, ly = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame, (lx, ly), 2, (0, 255, 255), -1)

        else:
            current_status["face_detected"] = False
            current_status["alert"] = None

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================= WEB SERVER & AUDIO JS =================
PAGE_HTML = """
<html>
<head>
    <title>Reflex Pro</title>
    <style>
        body { background: #111; color: white; text-align: center; font-family: monospace; }
        #vid { border: 2px solid #333; max-width: 100%; }
        #status { font-size: 24px; margin-top: 10px; color: #888; }
    </style>
</head>
<body>
    <h1>Reflex Research Pipeline</h1>
    <button onclick="startAudio()">🔊 ENABLE AUDIO</button>
    <br><br>
    <img id="vid" src="/video_feed">
    <div id="status">System Ready</div>

    <script>
        // Simple Beep Function (No external files needed)
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        
        function beep(freq = 600, duration = 100, type = 'square') {
            if (ctx.state === 'suspended') ctx.resume();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = type;
            osc.frequency.value = freq;
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            setTimeout(() => osc.stop(), duration);
        }

        function startAudio() {
            beep(400, 100); // Test beep
            document.getElementById('status').innerText = "Audio Active";
        }

        // Poll Python for status every 100ms
        setInterval(() => {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    const st = document.getElementById('status');
                    
                    if (!data.face) {
                        st.innerText = "No Face Detected";
                        st.style.color = "gray";
                        return;
                    }

                    if (data.alert === 'blink') {
                        st.innerText = "BLINK!";
                        st.style.color = "yellow";
                        beep(800, 50, 'sine');
                    } 
                    else if (data.alert === 'shake') {
                        st.innerText = "SHAKE!";
                        st.style.color = "red";
                        beep(200, 150, 'sawtooth');
                    }
                    else if (data.alert === 'left' || data.alert === 'right') {
                        st.innerText = "EYE MOVEMENT: " + data.alert.toUpperCase();
                        st.style.color = "magenta";
                        beep(600, 50, 'triangle');
                    }
                    else {
                        st.innerText = "Stable";
                        st.style.color = "lime";
                    }
                });
        }, 100);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(PAGE_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status_check():
    # Chrome calls this to see if it should beep
    return jsonify({
        "alert": current_status["alert"],
        "face": current_status["face_detected"]
    })

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        vs.stop()
