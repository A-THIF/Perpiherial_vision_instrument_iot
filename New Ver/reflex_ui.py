import cv2
import mediapipe as mp
import time
import math
import threading
import subprocess
import atexit
import signal
import os
import sys
import cv2
import mediapipe as mp
import time
import math
import threading
import subprocess
import atexit
import signal
import os
import sys
import numpy as np
from collections import deque
from flask import Flask, Response, render_template_string, jsonify

# ================= CONFIGURATION =================
STREAM_URL = "tcp://127.0.0.1:8888"
CENTER_BOX_SIZE = 180
ALIGN_THRESH = 60
HOLD_TIME_REQ = 3.0

# Sensitivity (0.15 = 15% deviation from center triggers alert)
SENSITIVITY = 0.15 

# ================= FLASK & STATE =================
app = Flask(__name__)

state = {
    "phase": "ALIGN",   # ALIGN -> HOLD -> ACTIVE
    "alert": None,      # 'left', 'right', 'blink' or None
    "hold_start": 0,
    "face_detected": False,
    "center_h": 0.0,    # Calibration value
    "calibrating": False,
    "ratio_h": 0.0      # Live raw value for UI bar
}

# Buffer for smoothing
h_buffer = deque(maxlen=5)

def cleanup_handler(sig, frame):
    print("\n[INFO] Cleaning up...")
    os.system("pkill rpicam-vid")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_handler)

# ================= CAMERA HARDWARE =================
subprocess.Popen([
    "rpicam-vid", "-t", "0", "--width", "640", "--height", "480",
    "--framerate", "30", "--codec", "mjpeg", "--inline", "--listen", "-o", STREAM_URL
])
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

# ================= AI LOGIC =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

IRIS = [474, 475, 476, 477]
CORNERS = [33, 133]

def get_horizontal_gaze(landmarks, width, height):
    # Get Coordinates
    iris = np.mean([(landmarks[i].x, landmarks[i].y) for i in IRIS], axis=0)
    left_corner = np.array([landmarks[33].x, landmarks[33].y])
    right_corner = np.array([landmarks[133].x, landmarks[133].y])

    # Eye Center & Width
    eye_center_x = (left_corner[0] + right_corner[0]) / 2.0
    eye_width = np.linalg.norm(left_corner - right_corner)

    # Normalized Deviation (-0.5 Left to +0.5 Right)
    dx = (iris[0] - eye_center_x) / eye_width
    return dx

def generate_frames():
    while True:
        frame = vs.read()
        if frame is None: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Draw Static Center Box
        box_color = (255, 0, 0) # Blue
        if state["phase"] == "ACTIVE": box_color = (100, 100, 100) # Dim it when active
        
        cv2.rectangle(frame, (cx - CENTER_BOX_SIZE//2, cy - CENTER_BOX_SIZE//2),
                      (cx + CENTER_BOX_SIZE//2, cy + CENTER_BOX_SIZE//2), box_color, 2)

        state["alert"] = None
        state["ratio_h"] = 0.0

        if results.multi_face_landmarks:
            state["face_detected"] = True
            lm = results.multi_face_landmarks[0].landmark
            
            # --- FACE BOX ---
            x_vals = [l.x * w for l in lm]
            y_vals = [l.y * h for l in lm]
            fx, fy = int(np.mean(x_vals)), int(np.mean(y_vals))
            
            # Alignment Check
            dist_from_center = math.hypot(fx - cx, fy - cy)
            is_aligned = dist_from_center < ALIGN_THRESH

            # Draw Face Box
            f_color = (0, 0, 255)
            if is_aligned: f_color = (0, 255, 0)
            x_min, x_max = int(min(x_vals)), int(max(x_vals))
            y_min, y_max = int(min(y_vals)), int(max(y_vals))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), f_color, 1)

            # --- GAZE TRACKING ---
            raw_h = get_horizontal_gaze(lm, w, h)
            h_buffer.append(raw_h)
            avg_h = sum(h_buffer) / len(h_buffer)

            if state["phase"] == "ALIGN":
                if is_aligned:
                    state["phase"] = "HOLD"
                    state["hold_start"] = time.time()
                    h_buffer.clear()

            elif state["phase"] == "HOLD":
                if not is_aligned:
                    state["phase"] = "ALIGN"
                else:
                    elapsed = time.time() - state["hold_start"]
                    rem = HOLD_TIME_REQ - elapsed
                    
                    # Auto Calibrate
                    state["center_h"] = avg_h
                    
                    if rem <= 0:
                        state["phase"] = "ACTIVE"
                    else:
                        # Draw Countdown on Face
                        cv2.putText(frame, f"{rem:.1f}", (fx-20, y_min-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            elif state["phase"] == "ACTIVE":
                if not is_aligned:
                    state["phase"] = "ALIGN"
                else:
                    # Calculate Deviation
                    diff_h = avg_h - state["center_h"]
                    state["ratio_h"] = diff_h # Send to UI

                    # Draw Eye Center
                    lx, ly = int(lm[473].x * w), int(lm[473].y * h)
                    cv2.circle(frame, (lx, ly), 4, (0, 255, 255), -1)

                    # Triggers
                    if diff_h < -SENSITIVITY:
                        state["alert"] = "right"
                    elif diff_h > SENSITIVITY:
                        state["alert"] = "left"

        else:
            state["face_detected"] = False
            state["phase"] = "ALIGN"

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================= DASHBOARD UI =================
PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Reflex Dashboard</title>
    <style>
        body { margin: 0; background: #0d0d0d; color: #eee; font-family: 'Segoe UI', sans-serif; height: 100vh; display: flex; overflow: hidden; }
        
        /* LEFT SIDE: VIDEO */
        .video-container { flex: 2; display: flex; justify-content: center; align-items: center; background: #000; border-right: 2px solid #333; }
        img { max-width: 95%; max-height: 95%; border: 2px solid #555; border-radius: 8px; }

        /* RIGHT SIDE: DASHBOARD */
        .dashboard { flex: 1; padding: 20px; display: flex; flex-direction: column; justify-content: space-between; }
        
        .header { text-align: center; border-bottom: 1px solid #333; padding-bottom: 20px; }
        h1 { margin: 0; font-size: 24px; color: #00d4ff; }
        
        .status-box { background: #1a1a1a; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px; height: 100px; display: flex; flex-direction: column; justify-content: center; }
        .label { font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .value { font-size: 32px; font-weight: bold; margin-top: 5px; }
        
        .alert-box { background: #111; border: 2px solid #333; height: 150px; display: flex; align-items: center; justify-content: center; border-radius: 10px; margin-top: 20px; transition: all 0.2s; }
        .alert-text { font-size: 40px; font-weight: 900; display: none; }
        
        .audio-btn { background: #333; color: white; border: none; padding: 15px; width: 100%; font-size: 16px; cursor: pointer; border-radius: 5px; margin-top: auto; }
        .audio-btn:hover { background: #444; }
        .audio-active { background: #008000 !important; }

        /* Progress Bar for Eyes */
        .bar-container { background: #333; height: 20px; border-radius: 10px; margin-top: 20px; position: relative; overflow: hidden; }
        .bar-center { position: absolute; left: 50%; width: 2px; height: 100%; background: white; }
        .bar-fill { height: 100%; background: #00d4ff; width: 50%; position: absolute; top: 0; transition: width 0.1s, left 0.1s; opacity: 0.6; }

    </style>
</head>
<body>

    <div class="video-container">
        <img src="/video_feed">
    </div>

    <div class="dashboard">
        <div class="header">
            <h1>REFLEX PRO</h1>
            <p>Vision Tracking System</p>
        </div>

        <div class="status-box">
            <div class="label">System Phase</div>
            <div class="value" id="phase-disp">ALIGN</div>
        </div>

        <div style="margin-top: 30px;">
            <div class="label" style="text-align: center;">Left / Right Balance</div>
            <div class="bar-container">
                <div class="bar-center"></div>
                <div class="bar-fill" id="eye-bar" style="left: 50%; width: 0%;"></div>
            </div>
        </div>

        <div class="alert-box" id="alert-panel">
            <div class="alert-text" id="alert-msg">LEFT</div>
        </div>

        <button class="audio-btn" onclick="toggleAudio()" id="audio-btn">🔇 ENABLE AUDIO</button>
    </div>

    <script>
        let ctx;
        let audioEnabled = false;
        let lastAlert = null;

        function toggleAudio() {
            if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
            ctx.resume().then(() => {
                audioEnabled = true;
                document.getElementById('audio-btn').innerText = "🔊 AUDIO ACTIVE";
                document.getElementById('audio-btn').classList.add("audio-active");
                beep(600, 100);
            });
        }

        function beep(freq, dur=150) {
            if (!audioEnabled) return;
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.frequency.value = freq;
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            setTimeout(() => osc.stop(), dur);
        }

        setInterval(() => {
            fetch('/status').then(r => r.json()).then(data => {
                
                // 1. Update Phase Text
                const pDiv = document.getElementById('phase-disp');
                pDiv.innerText = data.phase;
                if(data.phase === "ACTIVE") pDiv.style.color = "#00ff00";
                else if(data.phase === "HOLD") pDiv.style.color = "yellow";
                else pDiv.style.color = "white";

                // 2. Update Eye Bar
                // Range is approx -0.2 (Right) to +0.2 (Left)
                // We map this to percentage
                let val = data.ratio * 300; // Scale up
                const bar = document.getElementById('eye-bar');
                
                if (val > 0) { // Looking Left
                    bar.style.left = "50%";
                    bar.style.width = Math.min(val, 50) + "%";
                } else { // Looking Right
                    let w = Math.min(Math.abs(val), 50);
                    bar.style.left = (50 - w) + "%";
                    bar.style.width = w + "%";
                }

                // 3. Handle Alerts
                const panel = document.getElementById('alert-panel');
                const txt = document.getElementById('alert-msg');

                if (data.phase === "ACTIVE" && data.alert) {
                    txt.style.display = "block";
                    txt.innerText = data.alert.toUpperCase();
                    
                    if (data.alert === "left") {
                        panel.style.background = "linear-gradient(90deg, #550055, black)";
                        panel.style.borderColor = "#ff00ff";
                        txt.style.color = "#ff00ff";
                    } else {
                        panel.style.background = "linear-gradient(90deg, black, #550055)";
                        panel.style.borderColor = "#ff00ff";
                        txt.style.color = "#ff00ff";
                    }

                    if (data.alert !== lastAlert) {
                        beep(data.alert === "left" ? 600 : 800);
                        lastAlert = data.alert;
                    }
                } else {
                    txt.style.display = "none";
                    panel.style.background = "#111";
                    panel.style.borderColor = "#333";
                    lastAlert = null;
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
def status():
    return jsonify({
        "phase": state["phase"],
        "alert": state["alert"],
        "ratio": state["ratio_h"]
    })

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        os.system("pkill rpicam-vid")
        vs.stop()
