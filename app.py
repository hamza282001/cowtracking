from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import urllib
import cv2
import base64
import numpy as np
import io
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('./best.pt')
class_list = ["cow"]
tracker = Tracker()

total_inference_time = 0
total_number_detections = 0
number_of_videos_tracked = 0
last_track = 0

app = Flask(__name__)
CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type' 

# Inference
@app.route('/inference', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def inference():

    tracker.id_count = 0
    tracker.center_points = {}
    
    file = request.get_json()
    f = urllib.request.urlopen(file["file"])
    myfile = f.read()
    vid_file = io.BytesIO(myfile)
    vid_file_name = f"video_file_detect.mp4"
    with open(vid_file_name, "wb") as vid_file_write:
        vid_file_write.write(vid_file.getbuffer())
        
    ids_list = []
    ids_list_check = []
    # frames_list = []
        
    cap = cv2.VideoCapture(vid_file_name)
    count = 0
    output_file = 'cowtest-output.mp4'
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        results = model.predict(frame)
        a = results[0].boxes.cpu().data.numpy()
        px = pd.DataFrame(a).astype("float")
        list = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if "cow" in c:
           # if "person" in c:
                list.append([x1, y1, x2, y2])
        bbox_id = []
        bbox_id = tracker.update(list)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2

            # People ID display
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            if id not in ids_list_check:
                ids_list_check.append(id)
                ids_list.append({"id":id})

        # Write the frame to the output video
        # frames_list.append({"id":base64.b64encode(frame.tobytes()).decode('utf-8')})
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the video capture and video writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    with open(output_file, "rb") as videoFile:
        video = base64.b64encode(videoFile.read()).decode("utf-8")

    print(ids_list)
    return jsonify({"video": video, "ids": ids_list})

@app.route('/track', methods=['POST'])
@cross_origin(supports_credentials=True)
def track():
    
    file = request.get_json()
    id_vid = file["id"]
    vid_file_name = f"video_file_detect.mp4"
        
    cap = cv2.VideoCapture(vid_file_name)
    count = 0
    output_file = 'cowtest-output-tracked.mp4'
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        results = model.predict(frame)
        a = results[0].boxes.cpu().data.numpy()
        px = pd.DataFrame(a).astype("float")
        list = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if "cow" in c:
           # if "person" in c:
                list.append([x1, y1, x2, y2])

        bbox_id = tracker.update(list)
        print(bbox_id)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            print("ID: ", id)
            print("Video ID: ", id_vid)
            if int(id) == int(id_vid):
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2
    
                # People ID display
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        # Write the frame to the output video
        # frames_list.append({"id":base64.b64encode(frame.tobytes()).decode('utf-8')})
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the video capture and video writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    with open(output_file, "rb") as videoFile:
        video = base64.b64encode(videoFile.read()).decode("utf-8")
    return jsonify({"video": video})


# Testing
@app.route('/testing', methods=['POST'])
def testing():
    return jsonify({"method":"done"})


