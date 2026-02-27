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
import random
import smbus2 as smbus
import RPi.GPIO as GPIO
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify

# ================= CONFIGURATION =================
STREAM_URL = "tcp://127.0.0.1:8888"
WIDTH, HEIGHT = 640, 480 
CENTER_BOX_SIZE = 160
ALIGN_THRESH = 60
HOLD_TIME_REQ = 2.0
SENSITIVITY = 0.15 

# Hardware Config
LED_PINS = [14, 15, 18, 23, 24, 25, 8, 7] 
MCP_ADDR = 0x20

# ================= GLOBAL STATE =================
app = Flask(__name__)

state = {
    "phase": "ALIGN",   # ALIGN -> HOLD -> ACTIVE
    "alert": None,
    "hold_start": 0,
    "center_h": 0.0,
    "ratio_h": 0.0,
    "game_running": False,
    "game_paused": False,
    "scores": {"correct": 0, "wrong": 0, "missed": 0},
    "current_target": -1,
    "game_log": []
}

# ================= HARDWARE CONTROLLER =================
class Hardware:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in LED_PINS:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        self.bus = None
        try:
            self.bus = smbus.SMBus(1)
            self.bus.write_byte_data(MCP_ADDR, 0x00, 0xFF) 
            self.bus.write_byte_data(MCP_ADDR, 0x0C, 0xFF) 
            print("[HW] MCP23017 Connected.")
        except: print("[HW] ERROR: MCP23017 Not Found!")

    def set_led(self, index, state):
        if 0 <= index < 8: GPIO.output(LED_PINS[index], GPIO.HIGH if state else GPIO.LOW)

    def clear_leds(self):
        for pin in LED_PINS: GPIO.output(pin, GPIO.LOW)

    def read_sensors(self):
        if not self.bus: return 0
        try: return (~self.bus.read_byte_data(MCP_ADDR, 0x12)) & 0xFF
        except: return 0

hw = Hardware()

# ================= GAME ENGINE =================
class GameEngine(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.active = False
        self.params = {}

    def start_game(self, direction, duration, led_time, notes):
        self.params = {
            "dir": direction,
            "dur": float(duration) * 60,
            "timeout": float(led_time),
            "notes": notes
        }
        state["scores"] = {"correct": 0, "wrong": 0, "missed": 0}
        self.active = True
        
    def run(self):
        while True:
            if not self.active:
                time.sleep(0.1)
                continue

            print("[GAME] Started")
            start_time = time.time()
            end_time = start_time + self.params["dur"]
            current_led = 0 
            
            while time.time() < end_time and self.active:
                # --- DEADMAN SWITCH (Check Face AND Eyes) ---
                if state["phase"] != "ACTIVE":
                    state["game_paused"] = True
                    hw.clear_leds() # PAUSE HARDWARE
                    print("[GAME] PAUSED - Unstable!")
                    
                    while state["phase"] != "ACTIVE" and self.active:
                        time.sleep(0.1)
                        end_time += 0.1 
                    
                    state["game_paused"] = False
                    print("[GAME] RESUMED")

                # 1. LED ON
                state["current_target"] = current_led
                hw.clear_leds()
                hw.set_led(current_led, True)
                
                # 2. Wait Input
                led_start = time.time()
                hit_type = "MISS"
                
                while (time.time() - led_start) < self.params["timeout"]:
                    if state["phase"] != "ACTIVE": break # Instant Pause Check

                    sensors = hw.read_sensors()
                    if sensors != 0:
                        print(f"[DEBUG] Sensors: {bin(sensors)}") 
                        if sensors & (1 << current_led): hit_type = "HIT"
                        else: hit_type = "WRONG"
                        break 
                    time.sleep(0.01)
                
                if state["phase"] != "ACTIVE": continue # Restart loop if paused

                hw.clear_leds()
                
                if hit_type == "HIT": state["scores"]["correct"] += 1
                elif hit_type == "WRONG": state["scores"]["wrong"] += 1
                else: state["scores"]["missed"] += 1

                if self.params["dir"] == "clock": current_led = (current_led + 1) % 8
                elif self.params["dir"] == "anti": current_led = (current_led - 1) % 8
                else: current_led = random.randint(0, 7)
                
                time.sleep(1.0)

            self.active = False
            state["game_running"] = False
            hw.clear_leds()
            print("[GAME] Finished")

game = GameEngine()
game.start()

# ================= CLEANUP =================
def cleanup_handler(sig, frame):
    hw.clear_leds()
    os.system("pkill rpicam-vid")
    sys.exit(0)
signal.signal(signal.SIGINT, cleanup_handler)

subprocess.Popen([
    "rpicam-vid", "-t", "0", 
    "--width", str(WIDTH), "--height", str(HEIGHT), 
    "--framerate", "20", 
    "--codec", "mjpeg", "--inline", "--listen", "-o", STREAM_URL
])
time.sleep(2.0)

# ================= VISION SYSTEM =================
vs = cv2.VideoCapture(STREAM_URL)
vs.set(cv2.CAP_PROP_BUFFERSIZE, 1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

IRIS = [474, 475, 476, 477]
CORNERS = [33, 133]

def get_gaze(lm, w):
    iris = np.mean([(lm[i].x * w, lm[i].y) for i in IRIS], axis=0)
    left = np.array([lm[33].x * w, lm[33].y])
    right = np.array([lm[133].x * w, lm[133].y])
    width = np.linalg.norm(left - right)
    return (iris[0] - (left[0]+right[0])/2) / width

def generate_frames():
    avg_h_list = []
    
    while True:
        success, frame = vs.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        cv2.rectangle(frame, (cx - CENTER_BOX_SIZE//2, cy - CENTER_BOX_SIZE//2),
                      (cx + CENTER_BOX_SIZE//2, cy + CENTER_BOX_SIZE//2), (255, 0, 0), 2)

        state["alert"] = None

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            fx, fy = int(np.mean([l.x*w for l in lm])), int(np.mean([l.y*h for l in lm]))
            is_aligned = math.hypot(fx-cx, fy-cy) < ALIGN_THRESH
            
            f_col = (0, 0, 255)
            if is_aligned: f_col = (0, 255, 0)
            
            x_min = int(min([l.x*w for l in lm]))
            y_min = int(min([l.y*h for l in lm]))
            x_max = int(max([l.x*w for l in lm]))
            y_max = int(max([l.y*h for l in lm]))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), f_col, 2)

            for idx in IRIS + CORNERS:
                px, py = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame, (px, py), 3, (255, 255, 0), -1)

            if state["phase"] == "ALIGN":
                if is_aligned:
                    state["phase"] = "HOLD"
                    state["hold_start"] = time.time()
                    avg_h_list = []
            
            elif state["phase"] == "HOLD":
                if not is_aligned:
                    state["phase"] = "ALIGN"
                else:
                    rem = HOLD_TIME_REQ - (time.time() - state["hold_start"])
                    avg_h_list.append(get_gaze(lm, w))
                    if rem <= 0:
                        state["center_h"] = sum(avg_h_list)/len(avg_h_list)
                        state["phase"] = "ACTIVE"
                    else:
                        cv2.putText(frame, f"HOLD: {rem:.1f}", (fx, y_min-10), 0, 0.8, (0,255,255), 2)

            elif state["phase"] == "ACTIVE":
                # 1. Check Face Stability
                if not is_aligned:
                    state["phase"] = "ALIGN"
                    state["alert"] = "lost"
                else:
                    # 2. Check Eye Stability
                    curr = get_gaze(lm, w)
                    diff = curr - state["center_h"]
                    state["ratio_h"] = diff
                    
                    if diff < -SENSITIVITY: 
                        state["alert"] = "right"
                        state["phase"] = "ALIGN" # Force Pause
                    elif diff > SENSITIVITY: 
                        state["alert"] = "left"
                        state["phase"] = "ALIGN" # Force Pause

        else:
            state["phase"] = "ALIGN"

        if state["game_paused"]:
            cv2.putText(frame, "PAUSED - UNSTABLE!", (160, 240), 0, 1.0, (0,0,255), 3)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================= WEB SERVER =================
PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Reflex V3</title>
    <style>
        body { margin: 0; background: #111; color: #eee; font-family: sans-serif; display: flex; height: 100vh; }
        .col-video { flex: 2; background: #000; display: flex; justify-content: center; align-items: center; border-right: 2px solid #333; }
        .col-ui { flex: 1; padding: 20px; display: flex; flex-direction: column; overflow-y: auto; }
        img { width: 100%; max-width: 640px; border: 2px solid #555; }
        .box { background: #222; padding: 15px; margin-bottom: 15px; border-radius: 8px; border: 1px solid #444; }
        h2 { margin-top: 0; color: #00d4ff; }
        input, select, textarea { width: 90%; background: #333; color: white; border: 1px solid #555; padding: 8px; margin: 5px 0; border-radius: 4px; }
        button { width: 100%; padding: 15px; font-weight: bold; font-size: 16px; cursor: pointer; background: #00d4ff; border: none; border-radius: 5px; margin-top: 10px; }
        .score-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 5px; text-align: center; }
        .score-val { font-size: 24px; font-weight: bold; }
        .green { color: #00ff00; } .red { color: #ff0000; } .gray { color: #888; }
        .blink-red { animation: blink 1s infinite; background-color: #500; }
        @keyframes blink { 50% { opacity: 0.7; } }
    </style>
</head>
<body>
    <div class="col-video"><img src="/video_feed"></div>
    
    <div class="col-ui">
        <div id="alert-box" class="box blink-red" style="display:none; text-align:center;">
            <h1>⚠️ PAUSED</h1>
            <p>EYES / FACE UNSTABLE</p>
        </div>

        <div class="box">
            <h2>Status: <span id="status-text">ALIGN</span></h2>
            <div style="height:10px; background:#444; border-radius:5px; overflow:hidden;">
                <div id="gaze-bar" style="width:50%; height:100%; background:#00d4ff; position:relative; left:25%; transition:0.1s;"></div>
            </div>
        </div>

        <div id="setup-panel" class="box">
            <h2>Patient Setup</h2>
            <textarea id="notes" placeholder="Patient Notes"></textarea>
            <label>Mode:</label>
            <select id="dir"><option value="clock">Clockwise</option><option value="anti">Anti-Clock</option><option value="random">Random</option></select>
            <label>Duration (Mins):</label>
            <input type="number" id="dur" value="5">
            <label>LED Timeout (Sec):</label>
            <input type="number" id="timeout" value="5">
            <button onclick="startGame()">START GAME</button>
        </div>

        <div id="score-panel" class="box" style="display:none;">
            <h2>Live Score</h2>
            <div class="score-grid">
                <div><div class="score-val green" id="s-cor">0</div>Hit</div>
                <div><div class="score-val red" id="s-wro">0</div>Bad</div>
                <div><div class="score-val gray" id="s-mis">0</div>Miss</div>
            </div>
            <button onclick="stopGame()" style="background:#ff0000;">STOP & RESET</button>
        </div>
        
        <button onclick="startAudio()">🔊 ENABLE AUDIO</button>
    </div>

    <script>
        let ctx;
        function startAudio() { ctx = new (window.AudioContext || window.webkitAudioContext)(); ctx.resume(); beep(400, 50); }
        function beep(freq, dur) {
            if (!ctx) return;
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.frequency.value = freq;
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            setTimeout(() => osc.stop(), dur);
        }

        function startGame() {
            fetch('/start_game', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    dir: document.getElementById('dir').value,
                    dur: document.getElementById('dur').value,
                    timeout: document.getElementById('timeout').value,
                    notes: document.getElementById('notes').value
                })
            }).then(() => {
                document.getElementById('setup-panel').style.display = 'none';
                document.getElementById('score-panel').style.display = 'block';
            });
        }

        function stopGame() {
            fetch('/stop_game');
            window.location.reload();
        }

        setInterval(() => {
            fetch('/status').then(r => r.json()).then(data => {
                document.getElementById('status-text').innerText = data.phase;
                const bar = document.getElementById('gaze-bar');
                bar.style.left = ((data.ratio * 200) + 25) + "%";

                if (data.game_running) {
                    document.getElementById('s-cor').innerText = data.scores.correct;
                    document.getElementById('s-wro').innerText = data.scores.wrong;
                    document.getElementById('s-mis').innerText = data.scores.missed;
                }

                if (data.game_running && data.phase !== "ACTIVE") {
                    document.getElementById('alert-box').style.display = 'block';
                    beep(200, 100); 
                } else {
                    document.getElementById('alert-box').style.display = 'none';
                }
            });
        }, 150);
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(PAGE_HTML)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status(): return jsonify(state)

@app.route('/start_game', methods=['POST'])
def start_game_route():
    data = request.json
    game.start_game(data['dir'], data['dur'], data['timeout'], data['notes'])
    state["game_running"] = True
    return "OK"

@app.route('/stop_game')
def stop_game_route():
    game.active = False
    state["game_running"] = False
    hw.clear_leds()
    return "OK"

if __name__ == '__main__':
    try: app.run(host='0.0.0.0', port=5000, threaded=True)
    except: pass
    finally:
        hw.clear_leds()
        os.system("pkill rpicam-vid")
