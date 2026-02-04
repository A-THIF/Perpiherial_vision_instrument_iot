from flask import Flask, render_template_string, Response
import cv2

app = Flask(__name__)

# Open Pi Camera (or USB camera if index 0)
camera = cv2.VideoCapture(0)

# Function to generate MJPEG frames
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for HTML page
@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head>
            <title>Raspberry Pi Camera</title>
        </head>
        <body style="text-align:center; background-color:black;">
            <h1 style="color:white;">📷 Raspberry Pi Camera Feed</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </body>
        </html>
    ''')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # host='0.0.0.0' allows access from other devices
    app.run(host='0.0.0.0', port=5000)
