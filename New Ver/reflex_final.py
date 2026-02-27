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
CENTER_BOX_SIZE = 180
ALIGN_THRESH = 60
HOLD_TIME_REQ = 3.0

# Sensitivity (Lower gap = More sensitive)
# 0.5 is center. 
HORIZ_THRESH_L = 0.60  # > 0.60 is Left
HORIZ_THRESH_R = 0.40  # < 0.40 is Right
VERT_THRESH_U = 0.38   # < 0.38 is Up
VERT_THRESH_D = 0.55   # > 0.55 is Down

# ================= FLASK & STATE =================
app = Flask(__name__)

state = {
    "phase": "ALIGN",
    "alert": None,
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
LEFT_EYE_CORNERS = [33, 133] # Outer, Inner
LEFT_EYE_LIDS = [159, 145]   # Top, Bottom

def get_gaze_ratios(landmarks, iris_idx, corners_idx, lids_idx, width, height):
    # Iris Center
    iris_pts = np.array([(landmarks[i].x * width, landmarks[i].y * height) for i in iris_idx])
    iris_center = np.mean(iris_pts, axis=0)
    
    # 1. Horizontal Ratio
    left = np.array([landmarks[corners_idx[0]].x * width, landmarks[corners_idx[0]].y * height])
    right = np.array([landmarks[corners_idx[1]].x * width, landmarks[corners_idx[1]].y * height])
    total_w = np.linalg.norm(right - left)
    dist_l = np.linalg.norm(iris_center - left)
    ratio_h = dist_l / total_w
    
    # 2. Vertical Ratio
    top = np.array([landmarks[lids_idx[0]].x * width, landmarks[lids_idx[0]].y * height])
    bot = np.array([landmarks[lids_idx[1]].x * width, landmarks[lids_idx[1]].y * height])
    total_h = np.linalg.norm(bot - top)
    dist_t = np.linalg.norm(iris_center - top)
    ratio_v = dist_t / total_h
    
    return ratio_h, ratio_v

def generate_frames():
    while True:
        frame = vs.read()
        if frame is None: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Static Center Box
        cv2.rectangle(frame, (cx - CENTER_BOX_SIZE//2, cy - CENTER_BOX_SIZE//2),
                      (cx + CENTER_BOX_SIZE//2, cy + CENTER_BOX_SIZE//2), (255, 0, 0), 2)

        msg = "ALIGN FACE"
        color = (0, 0, 255)
        state["alert"] = None

        if results.multi_face_landmarks:
            state["face_detected"] = True
            lm = results.multi_face_landmarks[0].landmark
            
            # --- FACE ALIGNMENT ---
            x_vals = [l.x * w for l in lm]
            y_vals = [l.y * h for l in lm]
            fx, fy = int(np.mean(x_vals)), int(np.mean(y_vals))
            
            face_color = (0, 0, 255)
            if state["phase"] == "ACTIVE": face_color = (0, 255, 0)
            
            # Draw Face Box
            x_min, x_max = int(min(x_vals)), int(max(x_vals))
            y_min, y_max = int(min(y_vals)), int(max(y_vals))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), face_color, 2)

            dist = math.hypot(fx - cx, fy - cy)
            is_aligned = dist < ALIGN_THRESH

            if state["phase"] == "ALIGN":
                if is_aligned:
                    state["phase"] = "HOLD"
                    state["hold_start"] = time.time()
                else:
                    msg = "MOVE TO BOX"

            elif state["phase"] == "HOLD":
                if not is_aligned:
                    state["phase"] = "ALIGN"
                else:
                    elapsed = time.time() - state["hold_start"]
                    rem = HOLD_TIME_REQ - elapsed
                    if rem <= 0:
                        state["phase"] = "ACTIVE"
                    else:
                        msg = f"HOLD: {rem:.1f}s"
                        color = (0, 255, 255)

            elif state["phase"] == "ACTIVE":
                msg = "ACTIVE"
                color = (0, 255, 0)
                
                if not is_aligned:
                    state["phase"] = "ALIGN"
                else:
                    # --- EYE TRACKING ---
                    # We only track LEFT eye landmarks for ratio (sufficient for gaze)
                    # Note: Because of Mirror Flip, "Left" eye on screen is your Right eye
                    # but logic holds.
                    ratio_h, ratio_v = get_gaze_ratios(lm, LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS, w, h)

                    # Visual Intensity Meters
                    # Horizontal Bar
                    bar_w = 200
                    cv2.rectangle(frame, (20, h-40), (20+bar_w, h-20), (50,50,50), -1)
                    pos_h = int(ratio_h * bar_w)
                    cv2.circle(frame, (20+pos_h, h-30), 8, (0,255,255), -1)
                    
                    # Vertical Bar
                    bar_h = 100
                    cv2.rectangle(frame, (w-40, h-120), (w-20, h-20), (50,50,50), -1)
                    pos_v = int(ratio_v * bar_h)
                    cv2.circle(frame, (w-30, h-120+pos_v), 8, (0,255,255), -1)

                    # Logic
                    if ratio_h < HORIZ_THRESH_R:
                        state["alert"] = "right"
                        cv2.putText(frame, "RIGHT >>", (w-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
                    elif ratio_h > HORIZ_THRESH_L:
                        state["alert"] = "left"
                        cv2.putText(frame, "<< LEFT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
                    elif ratio_v < VERT_THRESH_U:
                        state["alert"] = "up"
                        cv2.putText(frame, "^ UP ^", (cx-50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
                    elif ratio_v > VERT_THRESH_D:
                        state["alert"] = "down"
                        cv2.putText(frame, "v DOWN v", (cx-70, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

                    # Draw Eye Center
                    lx, ly = int(lm[473].x * w), int(lm[473].y * h)
                    cv2.circle(frame, (lx, ly), 3, (0, 0, 255), -1)

        else:
            state["face_detected"] = False
            state["phase"] = "ALIGN"

        cv2.putText(frame, msg, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================= JAVASCRIPT & HTML =================
PAGE_HTML = """
<html>
<head>
    <title>Reflex Final</title>
    <style>body { background: #111; color: white; text-align: center; font-family: sans-serif; }</style>
</head>
<body>
    <h2>Reflex Pro: Gaze Training</h2>
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
            alert("Audio Active!");
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
    finally:
        vs.stop()
