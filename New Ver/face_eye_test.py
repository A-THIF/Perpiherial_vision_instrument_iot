import cv2
import mediapipe as mp
import time
import math
import threading
import subprocess
import atexit
from flask import Flask, Response, jsonify, render_template

# ================= FLASK APP =================
app = Flask(__name__)

# ================= CONFIG =================
EYE_AR_THRESH = 0.22
CENTER_BOX_SIZE = 150
ALIGN_THRESHOLD = 50
HOLD_TIME_REQ = 3.0
EYE_TEST_TIME = 2.0
STREAM_URL = "tcp://127.0.0.1:8888"

# ================= START CAMERA AUTOMATICALLY =================
camera_process = subprocess.Popen([
    "rpicam-vid",
    "-t", "0",
    "--width", "640",
    "--height", "480",
    "--framerate", "30",
    "--codec", "mjpeg",
    "--listen",
    "-o", STREAM_URL
])
atexit.register(camera_process.terminate)
time.sleep(2)  # Give camera time to start

# ================= THREADED CAMERA =================
class CameraStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed:
                self.grabbed = True
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

vs = CameraStream(STREAM_URL).start()

# ================= MEDIAPIPE =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= STATE =================
state = "ALIGN"
hold_start_time = 0
eye_start_time = 0
blink_flag = False
shake_flag = False
prev_nose = None

# ================= EYE LANDMARKS =================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ================= UTIL =================
def calculate_ear(lm, indices):
    p2_p6 = math.hypot(lm[indices[1]].x - lm[indices[5]].x, lm[indices[1]].y - lm[indices[5]].y)
    p3_p5 = math.hypot(lm[indices[2]].x - lm[indices[4]].x, lm[indices[2]].y - lm[indices[4]].y)
    p1_p4 = math.hypot(lm[indices[0]].x - lm[indices[3]].x, lm[indices[0]].y - lm[indices[3]].y)
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def generate_frames():
    global state, hold_start_time, eye_start_time, blink_flag, shake_flag, prev_nose
    while True:
        frame = vs.read()
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        message = "ALIGN FACE"
        box_color = (255, 0, 0)
        blink_flag = False
        shake_flag = False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # Nose
            nx, ny = int(lm[1].x * w), int(lm[1].y * h)

            # Face bounding box
            x_min = int(min([p.x for p in lm]) * w)
            y_min = int(min([p.y for p in lm]) * h)
            x_max = int(max([p.x for p in lm]) * w)
            y_max = int(max([p.y for p in lm]) * h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw eye landmarks (thin)
            for idx in LEFT_EYE + RIGHT_EYE:
                ex, ey = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame, (ex, ey), 1, (0, 255, 255), -1)

            # Detect shake (nose movement)
            if prev_nose:
                dist_nose = math.hypot(nx - prev_nose[0], ny - prev_nose[1])
                if dist_nose > 25:  # shake threshold
                    shake_flag = True
            prev_nose = (nx, ny)

            # State machine
            dist_center = math.hypot(nx - cx, ny - cy)
            aligned = dist_center < ALIGN_THRESHOLD

            if state == "ALIGN":
                if aligned:
                    state = "HOLD"
                    hold_start_time = time.time()
                else:
                    box_color = (0, 0, 255)
            elif state == "HOLD":
                if not aligned:
                    state = "ALIGN"
                else:
                    remain = HOLD_TIME_REQ - (time.time() - hold_start_time)
                    if remain <= 0:
                        state = "EYES"
                        eye_start_time = time.time()
                    else:
                        message = f"HOLD: {remain:.1f}s"
                        box_color = (0, 255, 255)
            elif state == "EYES":
                ear = (calculate_ear(lm, LEFT_EYE) + calculate_ear(lm, RIGHT_EYE)) / 2
                if ear < EYE_AR_THRESH:
                    message = "BLINK!"
                    box_color = (0, 0, 255)
                    blink_flag = True
                    eye_start_time = time.time()
                else:
                    elapsed = time.time() - eye_start_time
                    if elapsed > EYE_TEST_TIME:
                        state = "SUCCESS"
                    else:
                        message = f"STARE: {EYE_TEST_TIME - elapsed:.1f}s"
                        box_color = (255, 0, 255)
            elif state == "SUCCESS":
                message = "PASSED"
                box_color = (0, 255, 0)
                if time.time() - eye_start_time > 3:
                    state = "ALIGN"

            # Nose dot
            cv2.circle(frame, (nx, ny), 5, box_color, -1)

        # Corner alert
        alert_text = ""
        if blink_flag:
            alert_text = "BLINK!"
        elif shake_flag:
            alert_text = "SHAKE!"

        if alert_text:
            cv2.putText(frame, alert_text, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame
        ret, buf = cv2.imencode(".jpg", frame)
        frame_bytes = buf.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")

# ================= FLASK ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return jsonify({"blink": blink_flag, "shake": shake_flag})

# ================= MAIN =================
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        vs.stop()
