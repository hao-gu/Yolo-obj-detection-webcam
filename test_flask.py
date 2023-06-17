from flask import Flask, render_template, Response, request, jsonify
import cv2
import math
from ultralytics import YOLO

#model
model = YOLO("yolo-Weights/yolov8n.pt")
classNames = ["person"]

app = Flask(__name__)


camera = cv2.VideoCapture(0)

def gen_frames():                                         
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

def generate_frames():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)

    while True:
        ret, frame= camera.read()
        results = model(frame, stream=True) 
        if not ret:
            break
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)