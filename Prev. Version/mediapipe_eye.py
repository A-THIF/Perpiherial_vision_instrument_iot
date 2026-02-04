from flask import Flask, Response, request
from picamera2 import Picamera2
import cv2
import os
import mediapipe as mp

app = Flask(__name__)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# MediaPipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # includes iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Allowed deviation in pixels for alignment
ALIGN_TOLERANCE = 30

def generate_frames(zoom_factor=1.0):
    while True:
        frame = picam2.capture_array()
        h, w, _ = frame.shape

        zoom_factor = max(1.0, min(zoom_factor, 4.0))
        new_h = int(h / zoom_factor)
        new_w = int(w / zoom_factor)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        cropped_frame = frame[top:top + new_h, left:left + new_w]
        zoomed_frame = cv2.resize(cropped_frame, (w, h))

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(zoomed_frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw all face landmarks as small dots
                for lm in face_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(zoomed_frame, (cx, cy), 1, (0, 255, 0), -1)

                # Eye landmark indexes (approx from MediaPipe FaceMesh)
                LEFT_EYE_IDX = list(range(33, 133))   # left eye region
                RIGHT_EYE_IDX = list(range(362, 463)) # right eye region

                # Draw circles for eye area
                for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                    lm = face_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(zoomed_frame, (cx, cy), 2, (0, 0, 255), -1)

        else:
            cv2.putText(zoomed_frame, "No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Convert to BGR for streaming
        zoomed_frame = cv2.cvtColor(zoomed_frame, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', zoomed_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    zoom = request.args.get('zoom', default=1.0, type=float)
    return Response(generate_frames(zoom),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <h1>Raspberry Pi Camera Stream with Face & Eye Landmarks</h1>
    <input type="range" id="zoomSlider" min="1" max="4" step="0.1" value="1" />
    <span id="zoomValue">1.0x</span>
    <br/>
    <img id="videoFeed" src="/video?zoom=1" width="640" />
    
    <script>
      const slider = document.getElementById('zoomSlider');
      const zoomValue = document.getElementById('zoomValue');
      const videoFeed = document.getElementById('videoFeed');
      
      slider.oninput = function() {
        zoomValue.innerText = this.value + 'x';
        videoFeed.src = '/video?zoom=' + this.value;
      }
    </script>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
