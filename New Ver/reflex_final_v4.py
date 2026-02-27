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

# LED BCM PINS (Physical 8,10,12,16,18,22,24,26)
LED_PINS = [14, 15, 18, 23, 24, 25, 8, 7] 
MCP_ADDR = 0x20

# ================= GLOBAL STATE =================
app = Flask(__name__)

state = {
    "phase": "ALIGN",   
    "is_stable": False,
    "ratio_h": 0.0,
    "center_h": 0.0,
    "game_status": "SETUP", # SETUP, RUNNING, PAUSED, FINISHED
    "scores": {"correct": 0, "wrong": 0, "missed": 0},
    "notes": "",
    "hardware_ok": True,
    "current_target": -1
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
        self.init_mcp()

    def init_mcp(self):
        try:
            self.bus = smbus.SMBus(1)
            # Set Port A as Input, Enable Pullups
            self.bus.write_byte_data(MCP_ADDR, 0x00, 0xFF) 
            self.bus.write_byte_data(MCP_ADDR, 0x0C, 0xFF) 
            state["hardware_ok"] = True
            print("[HW] MCP23017 at 0x20 is ONLINE")
        except:
            state["hardware_ok"] = False
            print("[HW] ERROR: MCP23017 not found")

    def set_led(self, index, val):
        if 0 <= index < 8: GPIO.output(LED_PINS[index], GPIO.HIGH if val else GPIO.LOW)

    def clear_leds(self):
        for pin in LED_PINS: GPIO.output(pin, GPIO.LOW)

    def read_sensors(self):
        try:
            # Active LOW: touched = 0. We invert so touched = 1.
            raw = self.bus.read_byte_data(MCP_ADDR, 0x12)
            state["hardware_ok"] = True
            return (~raw) & 0xFF
        except:
            state["hardware_ok"] = False
            return 0

hw = Hardware()

# ================= GAME ENGINE =================
class GameEngine(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.active = False

    def start_game(self, params):
        self.params = params
        state["scores"] = {"correct": 0, "wrong": 0, "missed": 0}
        state["notes"] = params.get('notes', '')
        state["game_status"] = "RUNNING"
        self.active = True
        
    def run(self):
        while True:
            if not self.active:
                time.sleep(0.1)
                continue

            current_led = 0 
            duration_sec = float(self.params['dur']) * 60
            end_time = time.time() + duration_sec
            
            while time.time() < end_time and self.active:
                # --- SAFETY HALT (Face/Eye Movement) ---
                if state["phase"] != "ACTIVE":
                    hw.clear_leds()
                    while state["phase"] != "ACTIVE" and self.active:
                        time.sleep(0.1)
                        end_time += 0.1 # Extend session
                    if not self.active: break

                # LED Task
                hw.set_led(current_led, True)
                led_start = time.time()
                hit_detected = False
                
                while (time.time() - led_start) < float(self.params['timeout']):
                    if state["phase"] != "ACTIVE": break
                    
                    val = hw.read_sensors()
                    if val != 0:
                        # Correct logic: LED 1 matches Sensor 1 (Bit 0)
                        if val & (1 << current_led):
                            state["scores"]["correct"] += 1
                        else:
                            state["scores"]["wrong"] += 1
                        hit_detected = True
                        break
                    time.sleep(0.01)
                
                if not hit_detected and state["phase"] == "ACTIVE":
                    state["scores"]["missed"] += 1

                hw.clear_leds()
                
                # Sequencing
                if self.params['dir'] == "clock": current_led = (current_led + 1) % 8
                elif self.params['dir'] == "anti": current_led = (current_led - 1) % 8
                else: current_led = random.randint(0, 7)
                time.sleep(1.0)

            self.active = False
            state["game_status"] = "FINISHED"

engine = GameEngine()
engine.start()

# ================= VISION LOGIC =================
def generate_frames():
    cap = cv2.VideoCapture(STREAM_URL)
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    avg_list = []

    while True:
        success, frame = cap.read()
        if not success: continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        cx, cy = WIDTH//2, HEIGHT//2

        cv2.rectangle(frame, (cx-80, cy-80), (cx+80, cy+80), (255,0,0), 2)
        
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            fx, fy = int(np.mean([l.x*WIDTH for l in lm])), int(np.mean([l.y*HEIGHT for l in lm]))
            is_aligned = math.hypot(fx-cx, fy-cy) < ALIGN_THRESH
            
            # Cyan dots to prove eye detection
            for i in [474, 475, 476, 477, 33, 133]:
                cv2.circle(frame, (int(lm[i].x*WIDTH), int(lm[i].y*HEIGHT)), 2, (0,255,255), -1)

            if state["phase"] == "ALIGN":
                state["is_stable"] = False
                if is_aligned:
                    state["phase"] = "HOLD"
                    state["hold_start"] = time.time()
                    avg_list = []
            elif state["phase"] == "HOLD":
                if not is_aligned: state["phase"] = "ALIGN"
                else:
                    rem = HOLD_TIME_REQ - (time.time() - state["hold_start"])
                    iris = np.mean([(lm[i].x*WIDTH, lm[i].y) for i in [474,475,476,477]], axis=0)
                    eye_w = abs(lm[33].x - lm[133].x) * WIDTH
                    avg_list.append((iris[0] - (lm[33].x+lm[133].x)*WIDTH/2) / eye_w)
                    if rem <= 0:
                        state["center_h"] = sum(avg_list)/len(avg_list)
                        state["phase"] = "ACTIVE"
                        state["is_stable"] = True
                    cv2.putText(frame, f"LOCKING: {rem:.1f}", (fx-40, fy-100), 0, 0.7, (0,255,255), 2)
            elif state["phase"] == "ACTIVE":
                iris = np.mean([(lm[i].x*WIDTH, lm[i].y) for i in [474,475,476,477]], axis=0)
                eye_w = abs(lm[33].x - lm[133].x) * WIDTH
                curr_h = (iris[0] - (lm[33].x+lm[133].x)*WIDTH/2) / eye_w
                diff = curr_h - state["center_h"]
                state["ratio_h"] = diff
                
                if not is_aligned or abs(diff) > SENSITIVITY:
                    state["phase"] = "ALIGN"
                    state["is_stable"] = False
                else:
                    state["is_stable"] = True
        else:
            state["phase"] = "ALIGN"
            state["is_stable"] = False

        ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

# ================= DASHBOARD UI =================
HTML_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>Reflex Clinical Master</title>
    <style>
        body { margin:0; background:#0a0a0a; color:#fff; font-family:sans-serif; display:flex; height:100vh; }
        .pane-vid { flex:2; display:flex; flex-direction:column; align-items:center; justify-content:center; background:#000; }
        .pane-ui { flex:1; padding:25px; background:#111; border-left:1px solid #333; overflow-y:auto; }
        img { width:95%; border:2px solid #444; border-radius:10px; }
        .box { background:#1a1a1a; padding:20px; border-radius:10px; border:1px solid #333; margin-bottom:20px; }
        .hw-error { color:#f00; font-weight:bold; animation:blink 1s infinite; }
        @keyframes blink { 50% { opacity:0; } }
        .score { font-size:32px; font-weight:800; text-align:center; }
        textarea { width:100%; height:80px; background:#000; color:#0f0; border:1px solid #444; padding:10px; }
        .btn { width:100%; padding:15px; font-weight:bold; border:none; border-radius:5px; cursor:pointer; font-size:16px; margin-top:10px; }
        .btn-start { background:#007bff; color:#fff; }
        .btn-stop { background:#dc3545; color:#fff; }
        .btn-gray { background:#444; color:#fff; }
    </style>
</head>
<body>
    <div class="pane-vid">
        <img src="/video_feed">
        <h2 id="vision-text" style="color:#00d4ff;">ALIGNING...</h2>
    </div>
    <div class="pane-ui">
        <div id="hw-alert" class="box hw-error" style="display:none;">⚠️ HARDWARE ERROR: CHECK MCP WIRING</div>

        <div id="view-setup">
            <div class="box">
                <h2>Clinical Setup</h2>
                <textarea id="notes" placeholder="Enter patient details/observations..."></textarea>
                <label>Mode:</label>
                <select id="dir" style="width:100%; padding:10px; background:#222; color:#fff; margin:10px 0;">
                    <option value="clock">Clockwise</option>
                    <option value="anti">Anti-Clockwise</option>
                    <option value="random">Random</option>
                </select>
                <div style="display:flex; gap:10px;">
                    <div>Dur(min): <input type="number" id="dur" value="5" style="width:40px;"></div>
                    <div>Gap(sec): <input type="number" id="timeout" value="5" style="width:40px;"></div>
                </div>
                <button class="btn btn-start" onclick="startGame()">START TESTING</button>
            </div>
        </div>

        <div id="view-game" style="display:none;">
            <div class="box">
                <h2 style="color:#0f0;">Session Active</h2>
                <div class="score" style="color:#0f0;">HIT: <span id="s-cor">0</span></div>
                <div class="score" style="color:#f00;">BAD: <span id="s-wro">0</span></div>
                <div class="score" style="color:#888;">MISS: <span id="s-mis">0</span></div>
                <button class="btn btn-stop" onclick="stopGame()">END SESSION</button>
            </div>
            <div id="paused-alert" class="box" style="background:#500; display:none; text-align:center;">
                <h3>⚠️ SYSTEM HALTED</h3>
                <p>ALIGN FACE/EYES TO RESUME</p>
            </div>
        </div>

        <div id="view-report" style="display:none;">
            <div class="box">
                <h2 style="color:#00d4ff;">Session Summary</h2>
                <p id="r-notes"></p>
                <p id="r-stats"></p>
                <button class="btn btn-start" onclick="downloadReport()">DOWNLOAD REPORT (.TXT)</button>
                <button class="btn btn-gray" onclick="location.reload()">NEW SESSION</button>
            </div>
        </div>
        
        <button class="btn btn-gray" onclick="initAudio()">🔊 INITIALIZE AUDIO</button>
    </div>

    <script>
        let audioCtx;
        function initAudio() { audioCtx = new AudioContext(); }
        function beep(f, d) { if(!audioCtx) return; let o=audioCtx.createOscillator(); o.frequency.value=f; o.connect(audioCtx.destination); o.start(); setTimeout(()=>o.stop(), d); }

        function startGame() {
            const data = { notes: document.getElementById('notes').value, dir: document.getElementById('dir').value, dur: document.getElementById('dur').value, timeout: document.getElementById('timeout').value };
            fetch('/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)})
            .then(() => { document.getElementById('view-setup').style.display='none'; document.getElementById('view-game').style.display='block'; });
        }

        function stopGame() { fetch('/stop'); }

        function downloadReport() {
            const notes = document.getElementById('notes').value;
            const content = `REFLEX TEST REPORT\\n\\nNotes: ${notes}\\nCorrect: ${document.getElementById('s-cor').innerText}\\nWrong: ${document.getElementById('s-wro').innerText}\\nMissed: ${document.getElementById('s-mis').innerText}`;
            const blob = new Blob([content], {type:'text/plain'});
            const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'report.txt'; a.click();
        }

        setInterval(() => {
            fetch('/status').then(r=>r.json()).then(data => {
                document.getElementById('vision-text').innerText = data.phase + (data.is_stable ? " (STABLE)" : "");
                document.getElementById('vision-text').style.color = data.is_stable ? "#0f0" : "#00d4ff";
                document.getElementById('hw-alert').style.display = data.hardware_ok ? 'none' : 'block';

                if (data.game_status === "RUNNING") {
                    document.getElementById('s-cor').innerText = data.scores.correct;
                    document.getElementById('s-wro').innerText = data.scores.wrong;
                    document.getElementById('s-mis').innerText = data.scores.missed;
                    document.getElementById('paused-alert').style.display = (!data.is_stable) ? 'block' : 'none';
                    if(!data.is_stable) beep(200, 50);
                } else if (data.game_status === "FINISHED") {
                    document.getElementById('view-game').style.display='none';
                    document.getElementById('view-report').style.display='block';
                    document.getElementById('r-notes').innerText = "Clinical Notes: " + data.notes;
                }
            });
        }, 200);
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML_UI)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status(): return jsonify(state)

@app.route('/start', methods=['POST'])
def start_game_route():
    engine.start_game(request.json)
    return "OK"

@app.route('/stop')
def stop_game_route():
    engine.active = False
    return "OK"

if __name__ == '__main__':
    subprocess.Popen(["rpicam-vid", "-t", "0", "--width", str(WIDTH), "--height", str(HEIGHT), "--framerate", "20", "--codec", "mjpeg", "--inline", "--listen", "-o", STREAM_URL])
    try: app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        hw.clear_leds()
        os.system("pkill rpicam-vid")
