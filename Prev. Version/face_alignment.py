from flask import Flask, Response, request
from picamera2 import Picamera2
import cv2
import os

app = Flask(__name__)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# Load Haar cascade from OpenCV's built-in path
face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
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

        gray = cv2.cvtColor(zoomed_frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            # Use first detected face
            (x, y, fw, fh) = faces[0]
            face_center = (x + fw // 2, y + fh // 2)

            # Draw green box for detected face
            cv2.rectangle(zoomed_frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            # Target center (middle of the screen)
            target_center = (w // 2, h // 2)

            # Check alignment
            dx = abs(face_center[0] - target_center[0])
            dy = abs(face_center[1] - target_center[1])
            aligned = dx <= ALIGN_TOLERANCE and dy <= ALIGN_TOLERANCE

            # Draw alignment box with same size as detected face
            color = (0, 0, 255) if aligned else (255, 255, 0)  # Red if aligned, sky blue otherwise
            cv2.rectangle(
                zoomed_frame,
                (target_center[0] - fw // 2, target_center[1] - fh // 2),
                (target_center[0] + fw // 2, target_center[1] + fh // 2),
                color, 2
            )

            # Optional: Text feedback
            if not aligned:
                cv2.putText(zoomed_frame, "Align face in the red box",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(zoomed_frame, "No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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
    <h1>Raspberry Pi Camera Stream with Face Alignment Guide</h1>
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
