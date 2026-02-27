import cv2
import mediapipe as mp
import time
import math
import threading
import subprocess
import atexit
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

# ================= CONFIGURATION =================
STREAM_URL = "tcp://127.0.0.1:8888"
CENTER_BOX_SIZE = 180   # Size of the static middle box
ALIGN_THRESH = 60       # How close (pixels) face must be to center
HOLD_TIME_REQ = 3.0     # Time to hold before unlocking eyes

# ================= FLASK & STATE =================
app = Flask(__name__)

# Global Status
state = {
    "phase": "ALIGN",   # ALIGN -> HOLD -> ACTIVE
    "alert": None,      # 'left', 'right', 'blink'
    "hold_start": 0,
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

# ================= AI LOGIC =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_CORNERS = [33, 133]
RIGHT_CORNERS = [362, 263]

def get_gaze_ratio(landmarks, iris_idx, corners_idx, width):
    iris_pts = np.array([(landmarks[i].x * width, landmarks[i].y) for i in iris_idx])
    iris_center = np.mean(iris_pts, axis=0)
    left = np.array([landmarks[corners_idx[0]].x * width, landmarks[corners_idx[0]].y])
    right = np.array([landmarks[corners_idx[1]].x * width, landmarks[corners_idx[1]].y])
    total_width = np.linalg.norm(right - left)
    dist_to_left = np.linalg.norm(iris_center - left)
    return dist_to_left / total_width

def generate_frames():
    while True:
        frame = vs.read()
        if frame is None: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Draw Static Center Box (Blue)
        cv2.rectangle(frame, (cx - CENTER_BOX_SIZE//2, cy - CENTER_BOX_SIZE//2),
                      (cx + CENTER_BOX_SIZE//2, cy + CENTER_BOX_SIZE//2), (255, 0, 0), 2)

        msg_text = "ALIGN FACE"
        msg_color = (0, 0, 255) # Red
        state["alert"] = None # Reset alert every frame

        if results.multi_face_landmarks:
            state["face_detected"] = True
            lm = results.multi_face_landmarks[0].landmark
            
            # --- 1. FACE BOX (Dynamic) ---
            x_vals = [l.x * w for l in lm]
            y_vals = [l.y * h for l in lm]
            fx, fy = int(np.mean(x_vals)), int(np.mean(y_vals)) # Face Center
            
            # Draw Face Box (Red if bad, Green if good)
            face_color = (0, 0, 255)
            if state["phase"] == "ACTIVE": face_color = (0, 255, 0)
            
            x_min, x_max = int(min(x_vals)), int(max(x_vals))
            y_min, y_max = int(min(y_vals)), int(max(y_vals))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), face_color, 2)

            # --- 2. ALIGNMENT CHECK ---
            dist_from_center = math.hypot(fx - cx, fy - cy)
            is_aligned = dist_from_center < ALIGN_THRESH

            # --- STATE MACHINE ---
            if state["phase"] == "ALIGN":
                if is_aligned:
                    state["phase"] = "HOLD"
                    state["hold_start"] = time.time()
                else:
                    msg_text = "MOVE TO BOX"

            elif state["phase"] == "HOLD":
                if not is_aligned:
                    state["phase"] = "ALIGN" # Failed, reset
                else:
                    elapsed = time.time() - state["hold_start"]
                    countdown = HOLD_TIME_REQ - elapsed
                    if countdown <= 0:
                        state["phase"] = "ACTIVE"
                    else:
                        msg_text = f"HOLD: {countdown:.1f}s"
                        msg_color = (0, 255, 255) # Yellow

            elif state["phase"] == "ACTIVE":
                msg_text = "ACTIVE - TRACKING EYES"
                msg_color = (0, 255, 0) # Green
                
                if not is_aligned:
                    state["phase"] = "ALIGN" # Lost face position
                else:
                    # --- 3. EYE TRACKING (Only in Active Mode) ---
                    # Draw Eye Landmarks
                    for idx in [33, 133, 362, 263, 468, 473]:
                        lx, ly = int(lm[idx].x * w), int(lm[idx].y * h)
                        cv2.circle(frame, (lx, ly), 2, (0, 255, 255), -1)

                    # Gaze Detection
                    r_ratio = get_gaze_ratio(lm, RIGHT_IRIS, RIGHT_CORNERS, w)
                    l_ratio = get_gaze_ratio(lm, LEFT_IRIS, LEFT_CORNERS, w)
                    avg_gaze = (r_ratio + l_ratio) / 2

                    # Sensitivity: < 0.45 Right, > 0.55 Left
                    if avg_gaze < 0.42:
                        state["alert"] = "right"
                        cv2.putText(frame, "RIGHT >>", (w-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)
                    elif avg_gaze > 0.58:
                        state["alert"] = "left"
                        cv2.putText(frame, "<< LEFT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)

        else:
            state["face_detected"] = False
            state["phase"] = "ALIGN"

        # Draw Status
        cv2.putText(frame, msg_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, msg_color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================= CLIENT SIDE AUDIO =================
PAGE_HTML = """
<html>
<head>
    <title>Reflex Final</title>
    <style>body { background: #111; color: white; text-align: center; font-family: sans-serif; }</style>
</head>
<body>
    <h2>Reflex Pro: Alignment Phase</h2>
    <button onclick="startAudio()" style="padding:10px; font-size:16px;">🔊 CLICK TO ENABLE AUDIO</button>
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
        alert("Audio enabled. Alerts will beep now.");
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
    setTimeout(() => osc.stop(), 150); // 150ms beep
}

// Poll server for alerts
setInterval(() => {
    if (!audioEnabled) return;
    fetch('/status')
        .then(r => r.json())
        .then(data => {
            if (data.phase !== "ACTIVE") return;

            if (data.alert && data.alert !== lastAlert) {
                if (data.alert === "left") beep(600);
                if (data.alert === "right") beep(800);
                lastAlert = data.alert;
            }
            else if (!data.alert) {
                lastAlert = null; // reset when no alert
            }
        });
}, 150);
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
    finally:
        vs.stop()
