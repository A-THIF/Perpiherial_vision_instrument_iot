from flask import Flask, Response, request, jsonify
from picamera2 import Picamera2
import cv2
import os
import mediapipe as mp
import time
import math
from collections import deque
import RPi.GPIO as GPIO

app = Flask(__name__)

# ==== CAMERA SETUP ====
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# ==== GPIO SETUP ====
LED_PIN = 17
SENSOR_POWER = 27
SENSOR_SIGNAL = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(SENSOR_POWER, GPIO.OUT)
GPIO.setup(SENSOR_SIGNAL, GPIO.IN)

led_on = False  # LED state

# ==== SCORE COUNTERS ====
correct_score = 0
missed_score = 0
wrong_score = 0

# ==== FACE ALIGNMENT ====
face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
)

# ==== MEDIAPIPE SETUP ====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# ==== CONSTANTS ====
ALIGN_TOLERANCE = 30
FACE_STEADY_SECONDS = 5
PUPIL_MOVEMENT_THRESHOLD = 8
MOVEMENT_CONFIRM_FRAMES = 3
COOLDOWN_FRAMES = 10
READY_DELAY = 5       # Wait before LED ON after good focus
LED_ON_DURATION = 5   # LED stays on if no hand detected

# ==== STATE ====
face_aligned_start_time = None
face_aligned = False
static_pupil_positions = None
eye_movement_detected = False
pupil_positions_history = deque(maxlen=MOVEMENT_CONFIRM_FRAMES)
movement_detected_frames = 0
non_movement_frames = 0
cooldown_counter = 0

ready_timer_start = None
led_on_time_start = None
led_state = 'off'  # 'off', 'ready', 'on'

# ==== HELPER FUNCTIONS ====
def set_led_sensor(state):
    global led_on
    GPIO.output(LED_PIN, GPIO.HIGH if state else GPIO.LOW)
    GPIO.output(SENSOR_POWER, GPIO.HIGH if state else GPIO.LOW)
    led_on = state

def pupil_center(landmarks, indices, w, h):
    xs = [landmarks[idx].x * w for idx in indices]
    ys = [landmarks[idx].y * h for idx in indices]
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def average_positions(positions):
    avg_left_x = sum([p[0][0] for p in positions]) / len(positions)
    avg_left_y = sum([p[0][1] for p in positions]) / len(positions)
    avg_right_x = sum([p[1][0] for p in positions]) / len(positions)
    avg_right_y = sum([p[1][1] for p in positions]) / len(positions)
    return ((int(avg_left_x), int(avg_left_y)), (int(avg_right_x), int(avg_right_y)))

# ==== FRAME GENERATION ====
def generate_frames(zoom_factor=1.0):
    global face_aligned_start_time, face_aligned, static_pupil_positions
    global eye_movement_detected, movement_detected_frames, non_movement_frames
    global cooldown_counter, correct_score, missed_score, wrong_score
    global ready_timer_start, led_on_time_start, led_state

    while True:
        frame = picam2.capture_array()
        h, w, _ = frame.shape

        zoom_factor = max(1.0, min(zoom_factor, 4.0))
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
        top, left = (h - new_h) // 2, (w - new_w) // 2
        cropped_frame = frame[top:top+new_h, left:left+new_w]
        zoomed_frame = cv2.resize(cropped_frame, (w, h))

        # FACE ALIGNMENT
        gray = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            (x, y, fw, fh) = faces[0]
            face_center = (x + fw // 2, y + fh // 2)
            target_center = (w // 2, h // 2)

            dx, dy = abs(face_center[0] - target_center[0]), abs(face_center[1] - target_center[1])
            aligned = dx <= ALIGN_TOLERANCE and dy <= ALIGN_TOLERANCE

            if aligned:
                if face_aligned_start_time is None:
                    face_aligned_start_time = time.time()
                elif (time.time() - face_aligned_start_time) >= FACE_STEADY_SECONDS:
                    if not face_aligned:
                        face_aligned = True
                        static_pupil_positions = None
                        eye_movement_detected = False
                        pupil_positions_history.clear()
                        movement_detected_frames = 0
                        non_movement_frames = 0
                        cooldown_counter = 0
                        ready_timer_start = None
                        led_on_time_start = None
                        led_state = 'off'
                else:
                    face_aligned = False
            else:
                face_aligned_start_time = None
                face_aligned = False
                static_pupil_positions = None
                eye_movement_detected = False
                pupil_positions_history.clear()
                movement_detected_frames = 0
                non_movement_frames = 0
                cooldown_counter = 0
                ready_timer_start = None
                led_on_time_start = None
                if led_state != 'off':
                    set_led_sensor(False)
                led_state = 'off'

            # PUPIL TRACKING
            if face_aligned:
                rgb_frame = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    fl = results.multi_face_landmarks[0]
                    LEFT_IRIS_IDX = [474, 475, 476, 477]
                    RIGHT_IRIS_IDX = [469, 470, 471, 472]

                    left_pupil = pupil_center(fl.landmark, LEFT_IRIS_IDX, w, h)
                    right_pupil = pupil_center(fl.landmark, RIGHT_IRIS_IDX, w, h)

                    pupil_positions_history.append((left_pupil, right_pupil))
                    if static_pupil_positions is None and len(pupil_positions_history) == MOVEMENT_CONFIRM_FRAMES:
                        static_pupil_positions = average_positions(pupil_positions_history)
                        eye_movement_detected = False
                        movement_detected_frames = 0
                        non_movement_frames = 0
                    elif len(pupil_positions_history) == MOVEMENT_CONFIRM_FRAMES:
                        avg_positions = average_positions(pupil_positions_history)
                        left_dist = distance(avg_positions[0], static_pupil_positions[0])
                        right_dist = distance(avg_positions[1], static_pupil_positions[1])
                        movement_now = left_dist > PUPIL_MOVEMENT_THRESHOLD or right_dist > PUPIL_MOVEMENT_THRESHOLD

                        if movement_now:
                            movement_detected_frames += 1
                            non_movement_frames = 0
                        else:
                            non_movement_frames += 1
                            movement_detected_frames = 0

                        if cooldown_counter == 0:
                            if movement_detected_frames >= MOVEMENT_CONFIRM_FRAMES:
                                eye_movement_detected = True
                                cooldown_counter = COOLDOWN_FRAMES
                            elif non_movement_frames >= MOVEMENT_CONFIRM_FRAMES:
                                eye_movement_detected = False
                                cooldown_counter = COOLDOWN_FRAMES
                        else:
                            cooldown_counter -= 1

            # LED logic synced with status
            if face_aligned and not eye_movement_detected:
                if led_state == 'off':
                    ready_timer_start = time.time()
                    led_state = 'ready'
                elif led_state == 'ready':
                    if (time.time() - ready_timer_start) >= READY_DELAY:
                        set_led_sensor(True)
                        led_on_time_start = time.time()
                        led_state = 'on'
                elif led_state == 'on':
                    hand_detected = (GPIO.input(SENSOR_SIGNAL) == 0)
                    elapsed = time.time() - led_on_time_start
                    if hand_detected:
                        correct_score += 1
                        set_led_sensor(False)
                        led_state = 'off'
                        ready_timer_start = None
                        led_on_time_start = None
                    elif elapsed >= LED_ON_DURATION:
                        missed_score += 1
                        set_led_sensor(False)
                        led_state = 'off'
                        ready_timer_start = None
                        led_on_time_start = None
                    if eye_movement_detected:
                        wrong_score += 1
                        set_led_sensor(False)
                        led_state = 'off'
                        ready_timer_start = None
                        led_on_time_start = None
            else:
                ready_timer_start = None
                led_on_time_start = None
                if led_state != 'off':
                    set_led_sensor(False)
                led_state = 'off'

        else:
            face_aligned = False
            eye_movement_detected = False
            if led_state != 'off':
                set_led_sensor(False)
            led_state = 'off'

        ret, buffer = cv2.imencode('.jpg', zoomed_frame)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==== ROUTES ====
@app.route('/video')
def video():
    zoom = request.args.get('zoom', default=1.0, type=float)
    return Response(generate_frames(zoom),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
    <title>Peripheral Vision & Reflex Test</title>
    <style>
      body { font-family: Arial, sans-serif; background: #f9f9f9; }
      .container { display: flex; align-items: flex-start; }
      .video-container { margin-right: 20px; }
      .info-container { background: white; padding: 15px; border-radius: 8px; }
      img { border: 2px solid #ccc; border-radius: 6px; }
      p { margin: 6px 0; }
    </style>
    </head>
    <body>
      <h1>Peripheral Vision & Reflex Test</h1>
      <div class="container">
        <div class="video-container">
          <img id="videoFeed" src="/video?zoom=1" width="640" />
          <br/>
          <label>Zoom:</label>
          <input type="range" id="zoomSlider" min="1" max="4" step="0.1" value="1" />
          <span id="zoomValue">1.0x</span>
        </div>
        <div class="info-container">
          <p>Correct: <span id="scoreCorrect">0</span></p>
          <p>Missed: <span id="scoreMissed">0</span></p>
          <p>Wrong: <span id="scoreWrong">0</span></p>
          <p>LED State: <span id="ledState">OFF</span></p>
          <p>Face Status: <span id="faceStatus">...</span></p>
          <p>Eye Status: <span id="eyeStatus">...</span></p>
        </div>
      </div>
      <script>
        const zoomSlider = document.getElementById('zoomSlider');
        const zoomValue = document.getElementById('zoomValue');
        const videoFeed = document.getElementById('videoFeed');
        const faceStatus = document.getElementById('faceStatus');
        const eyeStatus = document.getElementById('eyeStatus');
        zoomSlider.oninput = function(){
          zoomValue.innerText = this.value + 'x';
          videoFeed.src = '/video?zoom=' + this.value;
        }
        function fetchScore(){
          fetch('/score').then(r=>r.json()).then(data=>{
            document.getElementById('scoreCorrect').innerText = data.correct;
            document.getElementById('scoreMissed').innerText = data.missed;
            document.getElementById('scoreWrong').innerText = data.wrong;
            document.getElementById('ledState').innerText = data.led_state.toUpperCase();
            faceStatus.innerText = data.face_aligned ? 'Face Aligned - Stable' : 'Align your face in the box';
            eyeStatus.innerText = data.eye_movement_detected ? 'Eye Movement Detected!' : 'Eyes Steady - Good Focus';
          });
        }
        setInterval(fetchScore, 500);
      </script>
    </body>
    </html>
    '''

@app.route('/score')
def score():
    return jsonify({
        'correct': correct_score,
        'missed': missed_score,
        'wrong': wrong_score,
        'led_on': led_on,
        'face_aligned': face_aligned,
        'eye_movement_detected': eye_movement_detected,
        'led_state': led_state
    })

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        GPIO.cleanup()
