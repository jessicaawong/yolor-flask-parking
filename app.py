from concurrent.futures import process
from flask import Flask, render_template, url_for, redirect, Response
import os
import cv2
import subprocess

# from object_detection import *
from yolor2 import *

app = Flask(__name__)

UPLOAD_FOLDER = '/static/uploads/'
# VIDEO = VideoStreaming()

# initialize the camera from local webcam
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/display/<filename>')
def display_video(filename):
  return redirect(url_for('static', filename='src/' + filename), code=301)

# gak dipake
def gen_frames():
  while True:
    success, frame = camera.read()
    if not success:
      break
    else:
      # os.system('python detect.py --source 0')
      ret, buffer = cv2.imencode('.jpg', frame)
      frame = buffer.tobytes()
      # subprocess.run(['python', 'detect.py', '--source', frame], shell=True)
      yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
  # return Response(VIDEO.show(), mimetype='multipart/x-mixed-replace;boundary=frame')
  return Response(load_yolor_and_process_each_frame(True, 'VID20221031150502_rotated.mp4', True, 0.5, []), mimetype='multipart/x-mixed-replace;boundary=frame')