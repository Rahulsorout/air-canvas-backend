from flask import Flask
from flask_cors import CORS  # Import CORS
import subprocess

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins="http://localhost:5173")


process = None  # global to hold the air_canvas process

@app.route('/start', methods=['POST'])
def launch_air_canvas():
    global process
    if process is None:
        process = subprocess.Popen(["python", "air_canvas.py"])
        return "Air Canvas Started!"
    return "Air Canvas is already running."

@app.route('/stop', methods=['POST'])
def stop_air_canvas():
    global process
    if process is not None:
        process.terminate()
        process = None
        return "Air Canvas Stopped!"
    return "Air Canvas was not running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
