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

# Sensitivity Tuning (Higher = Less Sensitive)
SENSITIVITY_H = 0.15 
SENSITIVITY_V = 0.08  # Vertical movement is naturally smaller

# ================= FLASK & STATE =================
app = Flask(__name__)

state = {
    "phase": "ALIGN",
    "alert": None,
    "hold_start": 0,
    "face_detected": False,
    "center_h": 0.0,
    "center_v": 0.0,
    "calibrating": False
}

h_buffer = deque(maxlen=5)
v_buffer = deque(maxlen=5)

def cleanup_handler(sig, frame):
    print("\n[INFO] Cleaning up camera...")
    os.system("pkill rpicam-vid")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_handler)

subprocess.Popen([
    "rpicam-vid", "-t", "0", "--width", "640", "--height", "480",
    "--framerate", "30", "--codec", "mjpeg", "--inline", "--listen", "-o", STREAM_URL
])
time.sleep(2.0)

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

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks (Left Eye Only is sufficient for gaze)
IRIS = [474, 475, 476, 477]
CORNERS = [33, 133] # Outer, Inner
LIDS = [159, 145]   # Top, Bottom

def get_precise_gaze(landmarks, width, height):
    # 1. Get Coordinates
    iris = np.mean([(landmarks[i].x, landmarks[i].y) for i in IRIS], axis=0)
    left_corner = np.array([landmarks[33].x, landmarks[33].y])
    right_corner = np.array([landmarks[133].x, landmarks[133].y])
    top_lid = np.array([landmarks[159].x, landmarks[159].y])
    bot_lid = np.array([landmarks[145].x, landmarks[145].y])

    # 2. Eye Center Calculation (Geometric Center of Corners & Lids)
    eye_center_x = (left_corner[0] + right_corner[0]) / 2.0
    eye_center_y = (top_lid[1] + bot_lid[1]) / 2.0

    # 3. Eye Dimensions (Reference Scale)
    eye_width = np.linalg.norm(left_corner - right_corner)
    eye_height = np.linalg.norm(top_lid - bot_lid)

    # 4. Normalized Deviation
    # How far is Iris from Center (Normalized by Eye Size)
    dx = (iris[0] - eye_center_x) / eye_width  # -0.5 (Left) to +0.5 (Right)
    dy = (iris[1] - eye_center_y) / eye_height # -0.5 (Up) to +0.5 (Down)

    return dx, dy

def generate_frames():
    while True:
        frame = vs.read()
        if frame is None: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        cv2.rectangle(frame, (cx - CENTER_BOX_SIZE//2, cy - CENTER_BOX_SIZE//2),
                      (cx + CENTER_BOX_SIZE//2, cy + CENTER_BOX_SIZE//2), (255, 0, 0), 2)

        msg = "ALIGN FACE"
        color = (0, 0, 255)
        state["alert"] = None

        if results.multi_face_landmarks:
            state["face_detected"] = True
            lm = results.multi_face_landmarks[0].landmark
            
            # --- 1. FACE BOX ---
            x_vals = [l.x * w for l in lm]
            y_vals = [l.y * h for l in lm]
            fx, fy = int(np.mean(x_vals)), int(np.mean(y_vals))
            
            face_color = (0, 255, 0) if state["phase"] == "ACTIVE" else (0, 0, 255)
            x_min, x_max = int(min(x_vals)), int(max(x_vals))
            y_min, y_max = int(min(y_vals)), int(max(y_vals))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), face_color, 2)

            dist_from_center = math.hypot(fx - cx, fy - cy)
            is_aligned = dist_from_center < ALIGN_THRESH

            # --- 2. GET PRECISE GAZE ---
            raw_h, raw_v = get_precise_gaze(lm, w, h)
            
            # Smooth
            h_buffer.append(raw_h)
            v_buffer.append(raw_v)
            avg_h = sum(h_buffer) / len(h_buffer)
            avg_v = sum(v_buffer) / len(v_buffer)

            if state["phase"] == "ALIGN":
                if is_aligned:
                    state["phase"] = "HOLD"
                    state["hold_start"] = time.time()
                    h_buffer.clear()
                    v_buffer.clear()
                else:
                    msg = "MOVE TO BOX"

            elif state["phase"] == "HOLD":
                if not is_aligned:
                    state["phase"] = "ALIGN"
                else:
                    elapsed = time.time() - state["hold_start"]
                    rem = HOLD_TIME_REQ - elapsed
                    
                    # Continuous Calibration
                    state["center_h"] = avg_h
                    state["center_v"] = avg_v
                    
                    if rem <= 0:
                        state["phase"] = "ACTIVE"
                    else:
                        msg = f"CALIBRATING... {rem:.1f}s"
                        color = (0, 255, 255)

            elif state["phase"] == "ACTIVE":
                msg = "TRACKING"
                color = (0, 255, 0)
                
                if not is_aligned:
                    state["phase"] = "ALIGN"
                else:
                    # Deviation from Calibrated Center
                    diff_h = avg_h - state["center_h"]
                    diff_v = avg_v - state["center_v"]

                    # --- VISUAL BARS ---
                    # Horizontal
                    bar_w = 200
                    bx = 20
                    by = h - 30
                    cv2.rectangle(frame, (bx, by-10), (bx+bar_w, by+10), (50,50,50), -1)
                    # Scale factor 2.0 to make movement visible
                    off_h = int((diff_h * 3.0) * (bar_w/2)) 
                    dot_x = bx + (bar_w//2) + off_h
                    dot_x = max(bx, min(bx+bar_w, dot_x))
                    cv2.circle(frame, (dot_x, by), 8, (0,255,255), -1)

                    # Vertical
                    bar_h_len = 100
                    bvx = w - 40
                    bvy = h - 120
                    cv2.rectangle(frame, (bvx-10, bvy), (bvx+10, bvy+bar_h_len), (50,50,50), -1)
                    off_v = int((diff_v * 3.0) * (bar_h_len/2))
                    dot_y = bvy + (bar_h_len//2) + off_v
                    dot_y = max(bvy, min(bvy+bar_h_len, dot_y))
                    cv2.circle(frame, (bvx, dot_y), 8, (0,255,255), -1)

                    # --- TRIGGERS ---
                    # Note: Y-axis is inverted in image coords (Up is negative)
                    if diff_h < -SENSITIVITY_H:
                        state["alert"] = "right"
                        cv2.putText(frame, "RIGHT >>", (w-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)
                    elif diff_h > SENSITIVITY_H:
                        state["alert"] = "left"
                        cv2.putText(frame, "<< LEFT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)
                    elif diff_v < -SENSITIVITY_V:
                        state["alert"] = "up"
                        cv2.putText(frame, "^ UP ^", (cx-50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)
                    elif diff_v > SENSITIVITY_V:
                        state["alert"] = "down"
                        cv2.putText(frame, "v DOWN v", (cx-70, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)

                    # Draw Eye
                    lx, ly = int(lm[473].x * w), int(lm[473].y * h)
                    cv2.circle(frame, (lx, ly), 3, (0, 255, 255), -1)

        else:
            state["face_detected"] = False
            state["phase"] = "ALIGN"

        cv2.putText(frame, msg, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================= WEB SERVER =================
PAGE_HTML = """
<html>
<head>
    <title>Reflex Calibrated</title>
    <style>body { background: #111; color: white; text-align: center; font-family: sans-serif; }</style>
</head>
<body>
    <h2>Reflex Pro: Calibrated Tracker</h2>
    <button onclick="startAudio()" style="padding:15px; font-size:18px; border-radius:8px; cursor:pointer;">
        🔊 ENABLE AUDIO
    </button>
    <br><br>
    <img src="/video_feed" style="border: 2px solid #444; width: 640px;">
    
    <script>
    let ctx;
    let audioEnabled = false;
    let lastAlert = null;

    function startAudio() {
        ctx = new (window.AudioContext || window.webkitAudioContext)();
        ctx.resume().then(() => {
            audioEnabled = true;
            console.log("Audio Enabled ✅");
            alert("Audio Active! Look Left/Right/Up/Down to test.");
        });
    }

    function beep(freq) {
        if (!audioEnabled) return;
        if (ctx.state === 'suspended') ctx.resume();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.frequency.value = freq;
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        setTimeout(() => osc.stop(), 150); 
    }

    setInterval(() => {
        if (!audioEnabled) return;
        fetch('/status')
            .then(r => r.json())
            .then(data => {
                if (data.phase !== "ACTIVE") return;

                if (data.alert && data.alert !== lastAlert) {
                    if (data.alert === "left") beep(600);
                    if (data.alert === "right") beep(800);
                    if (data.alert === "up") beep(1000);
                    if (data.alert === "down") beep(400);
                    lastAlert = data.alert;
                } else if (!data.alert) {
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
    return jsonify(state)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        os.system("pkill rpicam-vid")
        vs.stop()
