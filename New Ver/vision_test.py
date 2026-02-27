import cv2
import mediapipe as mp
import time
from flask import Flask, render_template_string, Response

app = Flask(__name__)

# --- CAMERA SETUP (Global) ---
# We try to open the camera once.
cap = cv2.VideoCapture(0)
cap.set(3, 640) # Width
cap.set(4, 480) # Height
cap.set(cv2.CAP_PROP_FPS, 30) # Force 30 FPS

# --- AI SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # This enables detailed Eye landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def check_alignment(frame):
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    status = "NO FACE"
    color = (0, 0, 255) # Red
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 1. HEAD ALIGNMENT (Nose vs Ears)
        nose_x = landmarks[1].x
        left_ear_x = landmarks[234].x
        right_ear_x = landmarks[454].x
        
        center_head = (left_ear_x + right_ear_x) / 2
        diff_head = abs(nose_x - center_head)
        
        # 2. EYE STABILITY (Iris Landmarks)
        # Left Iris Center: 468, Right Iris Center: 473
        # If tracking is bad, these jitter.
        
        if diff_head < 0.08: # If head is straight
            status = "PERFECT ALIGNMENT"
            color = (0, 255, 0) # Green
            
            # Draw a box around the face
            cv2.rectangle(frame, (int(left_ear_x * width), int(height * 0.2)), 
                                 (int(right_ear_x * width), int(height * 0.8)), color, 2)
        else:
            status = "ALIGN HEAD"
            color = (0, 165, 255) # Orange

        # Draw Eye Markers
        left_eye = landmarks[468]
        right_eye = landmarks[473]
        cv2.circle(frame, (int(left_eye.x * width), int(left_eye.y * height)), 3, (0, 255, 255), -1)
        cv2.circle(frame, (int(right_eye.x * width), int(right_eye.y * height)), 3, (0, 255, 255), -1)

    return frame, status, color

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            # If camera freezes, try to restart it automatically
            print("Camera freeze! Restarting...")
            cap.release()
            time.sleep(1)
            cap.open(0)
            continue

        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        
        # Process AI
        frame, text, color = check_alignment(frame)
        
        # Draw Status Text
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Encode for Browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Slow down slightly to save CPU (Prevents overheating/crash)
        time.sleep(0.01)

@app.route('/')
def index():
    return """
    <html>
    <body style="background:black; color:white; text-align:center;">
        <h1>Vision Alignment Test</h1>
        <img src="/video_feed" style="border: 2px solid white; width:80%;">
        <p>Green = Ready | Orange = Fix Head | Red = No Face</p>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        # Threaded=True prevents one Chrome tab from blocking the camera
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        cap.release()
