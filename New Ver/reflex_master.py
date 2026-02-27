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
from collections import deque
from flask import Flask, Response, render_template_string, request, jsonify

# ================= CONFIGURATION =================
STREAM_URL = "tcp://127.0.0.1:8888"
CENTER_BOX_SIZE = 180
ALIGN_THRESH = 60
HOLD_TIME_REQ = 3.0
SENSITIVITY = 0.15 

# Hardware Config
LED_PINS = [14, 15, 18, 23, 24, 25, 8, 7] # Physical: 8,10,12,16,18,22,24,26
MCP_ADDR = 0x20

# ================= GLOBAL STATE =================
app = Flask(__name__)

state = {
    # Vision State
    "phase": "ALIGN", 
    "alert": None,
    "hold_start": 0,
    "center_h": 0.0,
    "ratio_h": 0.0,
    
    # Game State
    "game_running": False,
    "score_correct": 0,
    "score_wrong": 0,
    "score_missed": 0,
    "current_target": -1, # Which LED is ON (0-7)
    "game_log": []        # To store patient notes
}

# ================= HARDWARE CONTROLLER =================
class Hardware:
    def __init__(self):
        # Setup LEDs
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in LED_PINS:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
            
        # Setup MCP (Sensors)
        self.bus = smbus.SMBus(1)
        try:
            # Set Port A as Input (0xFF) with Pullups (0xFF)
            self.bus.write_byte_data(MCP_ADDR, 0x00, 0xFF) # IODIRA
            self.bus.write_byte_data(MCP_ADDR, 0x0C, 0xFF) # GPPUA
        except:
            print("[ERROR] MCP23017 not found! Check wiring.")

    def set_led(self, index, state):
        # Turn specific LED on/off
        if 0 <= index < 8:
            GPIO.output(LED_PINS[index], GPIO.HIGH if state else GPIO.LOW)

    def clear_leds(self):
        for pin in LED_PINS:
            GPIO.output(pin, GPIO.LOW)

    def read_sensors(self):
        try:
            # Read Port A. Pins are LOW (0) when triggered because of pullups.
            # We invert (~) so 1 means triggered.
            data = ~self.bus.read_byte_data(MCP_ADDR, 0x12) # GPIOA
            return data & 0xFF 
        except:
            return 0

hw = Hardware()

# ================= GAME ENGINE =================
class GameEngine(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.active = False
        self.params = {}

    def start_game(self, direction, duration, led_time, notes):
        self.params = {
            "dir": direction,       # 'clock', 'anti', 'random'
            "dur": float(duration) * 60, # Minutes to Seconds
            "timeout": float(led_time),  # Seconds per LED
            "notes": notes
        }
        state["game_log"].append(f"START: {notes}")
        state["score_correct"] = 0
        state["score_wrong"] = 0
        state["score_missed"] = 0
        self.active = True
        
    def run(self):
        while True:
            if not self.active:
                time.sleep(0.1)
                continue

            start_time = time.time()
            end_time = start_time + self.params["dur"]
            current_led = 0 
            
            while time.time() < end_time and self.active:
                # 1. Turn ON LED
                state["current_target"] = current_led
                hw.clear_leds()
                hw.set_led(current_led, True)
                
                # 2. Wait for Input (Loop for 'timeout' seconds)
                led_start = time.time()
                hit_type = "MISS" # Default
                
                while (time.time() - led_start) < self.params["timeout"]:
                    sensors = hw.read_sensors()
                    
                    if sensors != 0: # Something was touched
                        # Check if the Correct bit is set
                        if sensors & (1 << current_led):
                            hit_type = "HIT"
                        else:
                            hit_type = "WRONG"
                        break # Exit wait loop
                    
                    time.sleep(0.01)
                
                # 3. Process Result
                hw.clear_leds()
                state["current_target"] = -1
                
                if hit_type == "HIT":
                    state["score_correct"] += 1
                    print(f"Unit {current_led}: CORRECT")
                elif hit_type == "WRONG":
                    state["score_wrong"] += 1
                    print(f"Unit {current_led}: WRONG SENSOR")
                else:
                    state["score_missed"] += 1
                    print(f"Unit {current_led}: MISSED")

                # 4. Determine Next LED
                if self.params["dir"] == "clock":
                    current_led = (current_led + 1) % 8
                elif self.params["dir"] == "anti":
                    current_led = (current_led - 1) % 8
                else:
                    current_led = random.randint(0, 7)
                
                # Small delay between units
                time.sleep(1.0)

            # Game Over
            self.active = False
            state["game_running"] = False
            hw.clear_leds()
            print("GAME OVER")

game = GameEngine()
game.start()

# ================= CLEANUP HANDLER =================
def cleanup_handler(sig, frame):
    print("\n[INFO] System Shutdown.")
    hw.clear_leds()
    os.system("pkill rpicam-vid")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_handler)

# ================= CAMERA PROCESS =================
subprocess.Popen([
    "rpicam-vid", "-t", "0", "--width", "480", "--height", "240",
    "--framerate", "20", "--codec", "mjpeg", "--inline", "--listen", "-o", STREAM_URL
])
time.sleep(2.0)

# ================= VISION SYSTEM =================
# (Standard Threaded Camera + Mediapipe Code)
vs = cv2.VideoCapture(STREAM_URL)
vs.set(cv2.CAP_PROP_BUFFERSIZE, 1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
h_buffer = deque(maxlen=5)

IRIS = [474, 475, 476, 477]
CORNERS = [33, 133]

def get_gaze(lm, w):
    iris = np.mean([(lm[i].x * w, lm[i].y) for i in IRIS], axis=0)
    left = np.array([lm[33].x * w, lm[33].y])
    right = np.array([lm[133].x * w, lm[133].y])
    width = np.linalg.norm(left - right)
    return (iris[0] - (left[0]+right[0])/2) / width

def generate_frames():
    while True:
        success, frame = vs.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Draw Center Box
        color = (255, 0, 0)
        if state["phase"] == "ACTIVE": color = (100, 100, 100)
        cv2.rectangle(frame, (cx-90, cy-90), (cx+90, cy+90), color, 2)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # Face Alignment
            fx, fy = int(np.mean([l.x*w for l in lm])), int(np.mean([l.y*h for l in lm]))
            is_aligned = math.hypot(fx-cx, fy-cy) < ALIGN_THRESH
            
            # Draw Face Box
            f_col = (0, 255, 0) if is_aligned else (0, 0, 255)
            x_min = int(min([l.x*w for l in lm]))
            x_max = int(max([l.x*w for l in lm]))
            y_min = int(min([l.y*h for l in lm]))
            y_max = int(max([l.y*h for l in lm]))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), f_col, 1)

            # State Machine
            if state["phase"] == "ALIGN":
                if is_aligned:
                    state["phase"] = "HOLD"
                    state["hold_start"] = time.time()
                    h_buffer.clear()
            elif state["phase"] == "HOLD":
                if not is_aligned: state["phase"] = "ALIGN"
                else:
                    rem = HOLD_TIME_REQ - (time.time() - state["hold_start"])
                    # Auto Calibrate
                    h_buffer.append(get_gaze(lm, w))
                    state["center_h"] = sum(h_buffer)/len(h_buffer)
                    if rem <= 0: state["phase"] = "ACTIVE"
                    cv2.putText(frame, f"{rem:.1f}", (fx, y_min-10), 0, 1, (0,255,255), 2)
            elif state["phase"] == "ACTIVE":
                if not is_aligned: state["phase"] = "ALIGN"
                else:
                    # Gaze Tracking
                    curr = get_gaze(lm, w)
                    diff = curr - state["center_h"]
                    state["ratio_h"] = diff
                    
                    if diff < -SENSITIVITY: state["alert"] = "right"
                    elif diff > SENSITIVITY: state["alert"] = "left"
                    else: state["alert"] = None

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================= FLASK ROUTES =================
PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Reflex Master</title>
    <style>
        body { margin: 0; background: #111; color: #eee; font-family: sans-serif; display: flex; height: 100vh; }
        .col-video { flex: 2; background: #000; display: flex; justify-content: center; align-items: center; border-right: 2px solid #333; }
        .col-ui { flex: 1; padding: 20px; display: flex; flex-direction: column; overflow-y: auto; }
        img { max-width: 95%; max-height: 95%; border: 2px solid #555; }
        
        .box { background: #222; padding: 15px; margin-bottom: 15px; border-radius: 8px; border: 1px solid #444; }
        h2 { margin-top: 0; color: #00d4ff; }
        
        input, select, textarea { width: 100%; background: #333; color: white; border: 1px solid #555; padding: 8px; margin: 5px 0; border-radius: 4px; }
        button { width: 100%; padding: 15px; font-weight: bold; font-size: 16px; cursor: pointer; background: #00d4ff; border: none; border-radius: 5px; margin-top: 10px; }
        button:disabled { background: #555; cursor: not-allowed; }
        
        .score-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 5px; text-align: center; }
        .score-val { font-size: 24px; font-weight: bold; }
        .green { color: #00ff00; } .red { color: #ff0000; } .gray { color: #888; }
        
        #game-panel { display: none; } /* Hidden until Active */
    </style>
</head>
<body>
    <div class="col-video"><img src="/video_feed"></div>
    
    <div class="col-ui">
        <div class="box">
            <h2>System Status</h2>
            <div id="status-text" style="font-size: 20px;">ALIGN FACE</div>
            <div style="margin-top:10px; height:10px; background:#444; border-radius:5px; overflow:hidden;">
                <div id="gaze-bar" style="width:50%; height:100%; background:#00d4ff; position:relative; left:25%; transition:0.1s;"></div>
            </div>
        </div>

        <div id="setup-panel" class="box" style="display:none; border-color: #00ff00;">
            <h2>Patient Setup</h2>
            <textarea id="notes" placeholder="Patient Notes (Optional)"></textarea>
            
            <label>Mode:</label>
            <select id="dir">
                <option value="clock">Clockwise</option>
                <option value="anti">Anti-Clockwise</option>
                <option value="random">Random</option>
            </select>
            
            <label>Duration (Mins):</label>
            <input type="number" id="dur" value="5" min="1">
            
            <label>LED Timeout (Sec):</label>
            <input type="number" id="timeout" value="5" min="1">
            
            <button onclick="startGame()">START GAME</button>
        </div>

        <div id="score-panel" class="box" style="display:none;">
            <h2>Live Score</h2>
            <div class="score-grid">
                <div><div class="score-val green" id="s-cor">0</div>Correct</div>
                <div><div class="score-val red" id="s-wro">0</div>Wrong</div>
                <div><div class="score-val gray" id="s-mis">0</div>Missed</div>
            </div>
            <button onclick="stopGame()" style="background:#ff0000;">STOP GAME</button>
        </div>
    </div>

    <script>
        let gameRunning = false;

        function startGame() {
            const data = {
                dir: document.getElementById('dir').value,
                dur: document.getElementById('dur').value,
                timeout: document.getElementById('timeout').value,
                notes: document.getElementById('notes').value
            };
            fetch('/start_game', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            }).then(() => {
                document.getElementById('setup-panel').style.display = 'none';
                document.getElementById('score-panel').style.display = 'block';
                gameRunning = true;
            });
        }

        function stopGame() {
            fetch('/stop_game'); // You would implement this route
            location.reload();
        }

        setInterval(() => {
            fetch('/status').then(r => r.json()).then(data => {
                // Update Status Text
                const st = document.getElementById('status-text');
                st.innerText = data.phase;
                
                // Update Gaze Bar
                const bar = document.getElementById('gaze-bar');
                const pct = (data.ratio * 200) + 25; // Map approx range to bar
                bar.style.left = pct + "%";

                // Show Setup Panel ONLY if Active & Not Playing
                if (data.phase === "ACTIVE" && !data.game_running) {
                    document.getElementById('setup-panel').style.display = 'block';
                } else if (data.phase !== "ACTIVE" && !data.game_running) {
                    document.getElementById('setup-panel').style.display = 'none';
                }

                // Update Scores
                if (data.game_running) {
                    document.getElementById('s-cor').innerText = data.scores.correct;
                    document.getElementById('s-wro').innerText = data.scores.wrong;
                    document.getElementById('s-mis').innerText = data.scores.missed;
                }
            });
        }, 100);
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(PAGE_HTML)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify({
        "phase": state["phase"],
        "ratio": state["ratio_h"],
        "game_running": state["game_running"],
        "scores": {
            "correct": state["score_correct"],
            "wrong": state["score_wrong"],
            "missed": state["score_missed"]
        }
    })

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
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except: pass
    finally:
        hw.clear_leds()
        os.system("pkill rpicam-vid")
