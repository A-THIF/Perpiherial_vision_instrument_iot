import cv2
import mediapipe as mp
import time
import random
import smbus2 as smbus
import RPi.GPIO as GPIO
from flask import Flask, render_template_string, Response

# ==========================================
# PART 1: HARDWARE CONTROLLER
# ==========================================
class RackHardware:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self.led_pins = [14, 15, 18, 23, 24, 25, 8, 7]
        for pin in self.led_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        self.MCP_ADDR = 0x20
        self.bus = smbus.SMBus(1)
        self.GPIOA = 0x12
        self.IODIRA = 0x00
        self.GPPUA = 0x0C
        
        try:
            self.bus.write_byte_data(self.MCP_ADDR, self.IODIRA, 0xFF)
            self.bus.write_byte_data(self.MCP_ADDR, self.GPPUA, 0xFF)
        except:
            pass

    def turn_led_on(self, unit_index):
        for pin in self.led_pins:
            GPIO.output(pin, GPIO.LOW)
        if 0 <= unit_index < 8:
            GPIO.output(self.led_pins[unit_index], GPIO.HIGH)

    def turn_all_off(self):
        for pin in self.led_pins:
            GPIO.output(pin, GPIO.LOW)

    def check_sensor(self, unit_index):
        try:
            data = self.bus.read_byte_data(self.MCP_ADDR, self.GPIOA)
            return not (data & (1 << unit_index))
        except:
            return False

    def cleanup(self):
        self.turn_all_off()
        GPIO.cleanup()

# ==========================================
# PART 2: GLOBAL SETUP (The Fix)
# ==========================================
app = Flask(__name__)
hw = RackHardware()

# Initialize Camera & AI ONCE (Global)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Global Game State
game_state = {
    "mode": 1,          # 1=Clock, 2=Anti, 3=Random
    "current_unit": 0,
    "state": "WAIT",    # WAIT, DELAY, ACTIVE
    "delay_start": 0,
    "reflex_start": 0,
    "msg": "Waiting for Face..."
}

# ==========================================
# PART 3: WEB SERVER LOGIC
# ==========================================
PAGE_HTML = """
<html>
<head>
    <title>Reflex Vision</title>
    <style>
        body { background: #111; color: white; text-align: center; font-family: sans-serif; }
        img { border: 2px solid #555; max-width: 100%; border-radius: 8px; }
        .btn { padding: 15px 30px; margin: 10px; font-size: 18px; cursor: pointer; background: #444; color: white; border: none; border-radius: 5px; }
        .btn:hover { background: #666; }
        .active { background: #007bff; }
    </style>
</head>
<body>
    <h2>Reflex Vision Trainer</h2>
    <img src="{{ url_for('video_feed') }}">
    <br><br>
    <button class="btn" onclick="setMode(1)">Clockwise</button>
    <button class="btn" onclick="setMode(2)">Anti-Clockwise</button>
    <button class="btn" onclick="setMode(3)">Random</button>
    
    <script>
        function setMode(m) { fetch('/set_mode/' + m); }
    </script>
</body>
</html>
"""

def process_game_logic(is_aligned):
    st = game_state
    now = time.time()

    if st['state'] == "WAIT":
        hw.turn_all_off()
        if is_aligned:
            st['msg'] = "READY"
            st['state'] = "DELAY"
            st['delay_start'] = now
        else:
            st['msg'] = "ALIGN FACE"

    elif st['state'] == "DELAY":
        hw.turn_all_off()
        if not is_aligned:
            st['state'] = "WAIT"
        elif now - st['delay_start'] > 1.0:
            # Next Unit Calculation
            if st['mode'] == 1:   st['current_unit'] = (st['current_unit'] + 1) % 8
            elif st['mode'] == 2: st['current_unit'] = (st['current_unit'] - 1) % 8
            else:                 st['current_unit'] = random.randint(0, 7)
            
            hw.turn_led_on(st['current_unit'])
            st['state'] = "ACTIVE"
            st['reflex_start'] = now
            st['msg'] = "GO!"

    elif st['state'] == "ACTIVE":
        if not is_aligned:
            st['state'] = "WAIT"
            return

        if now - st['reflex_start'] > 5.0:
            st['msg'] = "MISSED!"
            st['state'] = "DELAY"
            st['delay_start'] = now
            hw.turn_all_off()
        
        elif hw.check_sensor(st['current_unit']):
            reaction = now - st['reflex_start']
            st['msg'] = f"HIT! {reaction:.2f}s"
            st['state'] = "DELAY"
            st['delay_start'] = now
            hw.turn_all_off()

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            # Try to reconnect if camera drops
            cap.release()
            cap.open(0)
            continue

        # Face Process
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        is_aligned = False
        color = (0, 0, 255)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            nose = lm[1].x
            mid = (lm[234].x + lm[454].x) / 2
            if abs(nose - mid) < 0.08:
                is_aligned = True
                color = (0, 255, 0)

        # Run Game Logic
        process_game_logic(is_aligned)

        # Draw UI
        cv2.putText(frame, f"STATUS: {game_state['msg']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if game_state['state'] == "ACTIVE":
            cv2.putText(frame, f"UNIT: {game_state['current_unit']+1}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(PAGE_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<int:mode>')
def set_mode(mode):
    game_state['mode'] = mode
    game_state['state'] = "WAIT" # Reset state on change
    return "OK"

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        hw.cleanup()
        cap.release()
