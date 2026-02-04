from flask import Flask, Response, request, jsonify
from picamera2 import Picamera2
import cv2, os, mediapipe as mp, time, math
from collections import deque
import RPi.GPIO as GPIO
import numpy as np  # For generating blank frames when preview is off

app = Flask(__name__)

# ==== CAMERA SETUP ====
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# ==== GPIO SETUP ====
led_pins = {
    1: 17,   # Channel 1 LED GPIO
    8: 26    # Channel 8 LED GPIO
}
sensor_power_pins = {
    1: 27,   # Channel 1 sensor power GPIO
    8: 4     # Channel 8 sensor power GPIO
}
sensor_signal_pins = {
    1: 22,   # Channel 1 sensor signal GPIO
    8: 21    # Channel 8 sensor signal GPIO
}

GPIO.setmode(GPIO.BCM)
for pin in led_pins.values():
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
for pin in sensor_power_pins.values():
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
for pin in sensor_signal_pins.values():
    GPIO.setup(pin, GPIO.IN)

led_on = False

# ==== SCORE COUNTERS ====
correct_score = 0
missed_score = 0
wrong_score = 0

# ==== FACE ALIGNMENT / MEDIAPIPE ====
face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# ==== CONSTANTS ====
ALIGN_TOLERANCE = 30
FACE_STEADY_SECONDS = 5
PUPIL_MOVEMENT_THRESHOLD = 8
MOVEMENT_CONFIRM_FRAMES = 3
COOLDOWN_FRAMES = 10
CHANNEL_ON_TIME = 7  # seconds each channel stays ON

# ==== STATE VARIABLES ====
face_aligned_start_time = None
face_aligned = False
static_pupil_positions = None
eye_movement_detected = False
pupil_positions_history = deque(maxlen=MOVEMENT_CONFIRM_FRAMES)
movement_detected_frames = 0
non_movement_frames = 0
cooldown_counter = 0

led_state = 'off'
current_channel = 1
channel_list = [1, 8]
channel_index = 0
channel_start_time = None

preview_enabled = True  # Toggle for camera preview streaming

# ==== HELPER FUNCTIONS ====
def set_channel(state, ch):
    # Turn off all channels first
    for cid in channel_list:
        GPIO.output(led_pins[cid], GPIO.LOW)
        GPIO.output(sensor_power_pins[cid], GPIO.LOW)
    # Turn on given channel if state is True
    if state:
        GPIO.output(led_pins[ch], GPIO.HIGH)
        GPIO.output(sensor_power_pins[ch], GPIO.HIGH)

def pupil_center(landmarks, indices, w, h):
    xs = [landmarks[idx].x * w for idx in indices]
    ys = [landmarks[idx].y * h for idx in indices]
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0]) ** 2 + (p1[1]-p2[1]) ** 2)

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
    global led_state, current_channel, channel_index, channel_start_time, preview_enabled

    while True:
        if not preview_enabled:
            # Send blank frame to keep connection alive while preview off
            blank_frame = 255 * np.ones((480, 640, 3), np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        frame = picam2.capture_array()
        h, w, _ = frame.shape
        zoom_factor = max(1.0, min(zoom_factor, 4.0))
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
        top, left = (h - new_h) // 2, (w - new_w) // 2
        cropped = frame[top:top+new_h, left:left+new_w]
        zoomed = cv2.resize(cropped, (w, h))

        gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        left_pupil = right_pupil = None  # for drawing pupil circles

        if len(faces) > 0:
            (x, y, fw, fh) = faces[0]
            face_center = (x + fw // 2, y + fh // 2)
            target_center = (w // 2, h // 2)
            dx, dy = abs(face_center[0] - target_center[0]), abs(face_center[1] - target_center[1])
            aligned = dx <= ALIGN_TOLERANCE and dy <= ALIGN_TOLERANCE

            box_color = (0, 255, 0) if aligned else (0, 0, 255)
            cv2.rectangle(zoomed, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.rectangle(zoomed,
                          (target_center[0] - fw // 2, target_center[1] - fh // 2),
                          (target_center[0] + fw // 2, target_center[1] + fh // 2),
                          box_color, 2)

            current_time = time.time()
            if aligned:
                if face_aligned_start_time is None:
                    face_aligned_start_time = current_time
                elif (current_time - face_aligned_start_time) >= FACE_STEADY_SECONDS:
                    if not face_aligned:
                        face_aligned = True
                        static_pupil_positions = None
                        pupil_positions_history.clear()
                        movement_detected_frames = 0
                        non_movement_frames = 0
                        cooldown_counter = 0
            else:
                face_aligned = False
                face_aligned_start_time = None
                static_pupil_positions = None
                pupil_positions_history.clear()
                movement_detected_frames = 0
                non_movement_frames = 0
                cooldown_counter = 0

            face_status_text = "Face Aligned - Stable" if face_aligned else "Align your face in the box"
            face_status_color = (0, 255, 0) if face_aligned else (0, 0, 255)
            cv2.putText(zoomed, face_status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, face_status_color, 2)

            if face_aligned:
                rgb_frame = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
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
                else:
                    cv2.putText(zoomed, "No face landmarks detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            eye_status_text = "Eyes Steady - Good Focus" if not eye_movement_detected else "Eye Movement Detected!"
            eye_status_color = (0, 255, 0) if not eye_movement_detected else (0, 0, 255)
            cv2.putText(zoomed, eye_status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_status_color, 2)

            if left_pupil and right_pupil:
                cv2.circle(zoomed, left_pupil, 5, (255, 0, 0), -1)
                cv2.circle(zoomed, right_pupil, 5, (0, 0, 255), -1)

            # CHANNEL LOGIC
            if face_aligned and not eye_movement_detected:
                if led_state == 'off':
                    current_channel = channel_list[channel_index]
                    set_channel(True, current_channel)
                    led_state = 'on'
                    channel_start_time = time.time()
                elif led_state == 'on':
                    hand_detected = (GPIO.input(sensor_signal_pins[current_channel]) == 0)
                    elapsed = time.time() - channel_start_time
                    if hand_detected:
                        correct_score += 1
                        set_channel(False, current_channel)
                        led_state = 'off'
                        channel_index = (channel_index + 1) % len(channel_list)
                    elif elapsed >= CHANNEL_ON_TIME:
                        missed_score += 1
                        set_channel(False, current_channel)
                        led_state = 'off'
                        channel_index = (channel_index + 1) % len(channel_list)
                    if eye_movement_detected:
                        wrong_score += 1
                        set_channel(False, current_channel)
                        led_state = 'off'
                        channel_index = (channel_index + 1) % len(channel_list)
            else:
                set_channel(False, current_channel)
                led_state = 'off'
        else:
            face_aligned = False
            eye_movement_detected = False
            set_channel(False, current_channel)
            led_state = 'off'

        ret, buffer = cv2.imencode('.jpg', zoomed)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==== ROUTES ====
@app.route('/video')
def video():
    zoom = request.args.get('zoom', default=1.0, type=float)
    return Response(generate_frames(zoom), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_preview', methods=['POST'])
def toggle_preview():
    global preview_enabled
    global correct_score, missed_score, wrong_score
    preview_enabled = not preview_enabled
    # Reset scores when preview is turned off
    if not preview_enabled:
        correct_score = 0
        missed_score = 0
        wrong_score = 0
    return jsonify({'preview_enabled': preview_enabled})


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Peripheral Vision & Reflex Test (Channel 1 & 8 Loop)</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f9f9f9; }
            .container { display: flex; align-items: flex-start; }
            .video-container { margin-right: 20px; }
            .info-container { background: white; padding: 15px; border-radius: 8px; }
            img { border: 2px solid #ccc; border-radius: 6px; }
            p { margin: 6px 0; }
            button { margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>Peripheral Vision & Reflex Test (Channel 1 & 8 Loop)</h1>
        <div class="container">
            <div class="video-container">
                <img id="videoFeed" src="/video?zoom=1" width="640" />
                <br/>
                <label>Zoom:</label>
                <input type="range" id="zoomSlider" min="1" max="4" step="0.1" value="1" />
                <span id="zoomValue">1.0x</span>
                <br/>
                <button onclick="togglePreview()">Toggle Preview ON/OFF</button>
            </div>
            <div class="info-container">
                <p>Correct: <span id="scoreCorrect">0</span></p>
                <p>Missed: <span id="scoreMissed">0</span></p>
                <p>Wrong: <span id="scoreWrong">0</span></p>
                <p>LED State: <span id="ledState">OFF</span></p>
                <p>Face Status: <span id="faceStatus">Align your face in the box</span></p>
                <p>Eye Status: <span id="eyeStatus"></span></p>
                <p>Current Channel: <span id="currentChannel">1</span></p>
            </div>
        </div>
        <script>
            const zoomSlider = document.getElementById('zoomSlider');
            const zoomValue = document.getElementById('zoomValue');
            const videoFeed = document.getElementById('videoFeed');
            const faceStatus = document.getElementById('faceStatus');
            const eyeStatus = document.getElementById('eyeStatus');
            const currentChannel = document.getElementById('currentChannel');

            zoomSlider.oninput = function() {
                zoomValue.innerText = this.value + 'x';
                videoFeed.src = '/video?zoom=' + this.value;
            };

            function fetchScore() {
                fetch('/score').then(res => res.json()).then(data => {
                    document.getElementById('scoreCorrect').innerText = data.correct;
                    document.getElementById('scoreMissed').innerText = data.missed;
                    document.getElementById('scoreWrong').innerText = data.wrong;
                    document.getElementById('ledState').innerText = data.led_state.toUpperCase();
                    faceStatus.innerText = data.face_aligned ? 'Face Aligned - Stable' : 'Align your face in the box';
                    eyeStatus.innerText = data.eye_movement_detected ? 'Eye Movement Detected!' : 'Eyes Steady - Good Focus';
                    currentChannel.innerText = data.current_channel;
                });
            }
            setInterval(fetchScore, 500);

            function togglePreview() {
                fetch('/toggle_preview', { method: 'POST' }).then(res => res.json()).then(data => {
                    console.log("Preview Enabled:", data.preview_enabled);
                    if (data.preview_enabled) {
                        // Reload video to restart streaming
                        videoFeed.src = '/video?zoom=' + zoomSlider.value;
                    } else {
                        // Show blank image when preview is off
                        videoFeed.src = '';
                    }
                });
            }
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
        'face_aligned': face_aligned,
        'eye_movement_detected': eye_movement_detected,
        'led_state': led_state,
        'current_channel': current_channel
    })

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        GPIO.cleanup()
        picam2.close()
        print("GPIO cleaned up and camera closed.")
