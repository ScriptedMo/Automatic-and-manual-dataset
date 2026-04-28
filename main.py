from flask import Flask, Response, render_template, jsonify, send_file, send_from_directory
import cv2
import os
import time
import threading
import mediapipe as mp
from ultralytics import YOLO

app = Flask(__name__)

# YOLOv8 model
model = YOLO("yolov8s.pt")

# Mediapipe face detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

# output folder
output_dir = "motion_images"
os.makedirs(output_dir, exist_ok=True)

camera_running = False
cap = None

frame_lock = threading.Lock()

detections_count = {
    "person": 0,
    "car": 0,
    "face": 0
}

# جلوگیری از عکس گرفتن چندباره
tracked_objects = {}
cooldown = 10  # seconds


def save_detection(frame, label, track_id):

    now = time.time()

    if track_id in tracked_objects:
        if now - tracked_objects[track_id] < cooldown:
            return

    tracked_objects[track_id] = now

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    filename = f"{label}_{track_id}_{timestamp}.jpg"

    path = os.path.join(output_dir, filename)

    cv2.imwrite(path, frame)

    detections_count[label] += 1


def generate_frames():

    global camera_running, cap

    while camera_running:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, cls, track_id in zip(boxes, classes, ids):

                label = model.names[int(cls)]

                if label not in ["person", "car", "truck", "bus", "motorcycle"]:
                    continue

                x1, y1, x2, y2 = map(int, box)

                obj = frame[y1:y2, x1:x2]

                save_detection(obj, label if label != "person" else "person", track_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, f"{label} ID:{int(track_id)}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

        # Face detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(rgb)

        if face_results.detections:

            for detection in face_results.detections:

                bbox = detection.location_data.relative_bounding_box

                h, w, _ = frame.shape

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                ww = int(bbox.width * w)
                hh = int(bbox.height * h)

                face = frame[y:y+hh, x:x+ww]

                timestamp = time.strftime("%Y%m%d_%H%M%S")

                filename = f"face_{timestamp}.jpg"

                cv2.imwrite(os.path.join(output_dir, filename), face)

                detections_count["face"] += 1

                cv2.rectangle(frame, (x, y), (x+ww, y+hh), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():

    if camera_running:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    return "camera stopped"


@app.route('/start_camera')
def start_camera():

    global camera_running, cap

    if not camera_running:

        cap = cv2.VideoCapture(0)

        camera_running = True

        return jsonify(status="started")

    return jsonify(status="already_running")


@app.route('/stop_camera')
def stop_camera():

    global camera_running

    camera_running = False

    return jsonify(status="stopped")


@app.route('/get_detections')
def get_detections():

    return jsonify(detections=detections_count)


@app.route('/view_images')
def view_images():

    images = os.listdir(output_dir)

    images.sort(reverse=True)

    return render_template("view_images.html", images=images)


@app.route('/motion_images/<filename>')
def serve_image(filename):

    return send_from_directory(output_dir, filename)


@app.route('/download_image/<filename>')
def download_image(filename):

    return send_file(os.path.join(output_dir, filename), as_attachment=True)


@app.route('/delete_image/<filename>', methods=['POST'])
def delete_image(filename):

    path = os.path.join(output_dir, filename)

    if os.path.exists(path):

        os.remove(path)

        return jsonify(status="deleted")

    return jsonify(status="not_found")


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=False)
