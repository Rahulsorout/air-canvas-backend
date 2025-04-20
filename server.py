from flask import Flask, Response
from flask_cors import CORS
import subprocess
import cv2

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins=[
    "http://localhost:5173",
    "https://air-canvas-frontend.vercel.app"
])

process = None  # global to hold the air_canvas process
video_capture = None  # global to hold the video capture object

def generate_video_feed():
    """Generate frames from the webcam and send them as an HTTP response."""
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # Encode the frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        # Convert the frame to a byte array and yield it as part of a multipart HTTP response
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/start', methods=['POST'])
def launch_air_canvas():
    global process, video_capture
    if process is None:
        process = subprocess.Popen(["python", "air_canvas.py"])  # Start the AI/ML process
        video_capture = cv2.VideoCapture(0)  # Start video capture (webcam)
        return "Air Canvas Started!"
    return "Air Canvas is already running."

@app.route('/stop', methods=['POST'])
def stop_air_canvas():
    global process, video_capture
    if process is not None:
        process.terminate()  # Stop the Air Canvas process
        process = None
    if video_capture is not None:
        video_capture.release()  # Release the video capture object
        video_capture = None
    return "Air Canvas Stopped!"

@app.route('/video_feed')
def video_feed():
    """Route for streaming the video feed."""
    if video_capture is None:
        return "No video feed available. Start Air Canvas first."
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
