from flask import Flask, render_template,request,jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
import torch
import time
from collections import defaultdict
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from database.attendance import mark_attendance
import threading
from database.attendance import view_attendance,review_attendance,update_attendance,get_attendance

device = torch.device("cpu")

# Initialize models once
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load face embeddings
def load_model(load_path='face_recognition_model.pth'):
    loaded_data = torch.load(load_path, map_location=device)
    return loaded_data['face_embeddings'], loaded_data['names']

face_embeddings, known_face_names = load_model()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# OpenCV camera setup (0 for default webcam, replace with IP camera URL if needed)
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

# Temporary storage for recognized students
recognized_students = {
    "client": set(),
    "server": set()
}

# Lock for thread safety
lock = threading.Lock()

def recognize_faces(frame_bytes, source):
    """Detect and recognize faces in a frame and store recognized students."""
    try:
        img_np = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, _ = mtcnn.detect(img_pil)
        if boxes is not None:
            for box in boxes:
                face_img = img_pil.crop(tuple(box))
                face_tensor = mtcnn(face_img, return_prob=False)

                if face_tensor is not None:
                    with torch.no_grad():
                        emb = resnet(face_tensor.unsqueeze(0).to(device)).cpu()

                    min_dist, identity = min(((torch.dist(emb, e).item(), n) for e, n in zip(face_embeddings, known_face_names)), default=(float('inf'), "Unknown"))

                    if min_dist < 0.8:
                        with lock:
                            recognized_students[source].add(identity)
                        check_and_mark_attendance(identity)

    except Exception as e:
        print(f"Error in recognize_faces: {e}")

def check_and_mark_attendance(identity):
    """Mark attendance only if the student appears in both client and server streams."""
    with lock:
        client_present = identity in recognized_students["client"]
        server_present = identity in recognized_students["server"]
        
        print(f"📝 Checking {identity}: client({client_present}), server({server_present})")

        if client_present and server_present:
            print(f"✅ {identity} marked present!")
            mark_attendance(identity)

            # Remove from sets after marking attendance
            recognized_students["client"].remove(identity)
            recognized_students["server"].remove(identity)

            # Emit attendance update to client
            socketio.emit('attendance_update', {'name': identity, 'status': 'Marked'})

        else:
            print(f"⏳ Waiting for {identity} to appear on both streams...")


@socketio.on('video_frame')
def handle_client_video(data):
    """Process video frame from client."""
    recognize_faces(data['frame'], "client")

def process_server_camera():
    """Continuously process frames from the server's own camera."""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame from camera.")
            time.sleep(1)
            continue

        # Convert frame to bytes and process
        _, buffer = cv2.imencode(".jpg", frame)
        recognize_faces(buffer.tobytes(), "server")

        time.sleep(0.1)  # Adjust for performance

# Start server camera processing in a separate thread
server_camera_thread = threading.Thread(target=process_server_camera, daemon=True)
server_camera_thread.start()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return render_template('video_page.html')

@app.route('/view_marked')
def view_marked():
    attendance_records = view_attendance()
   
    return render_template('view_marked.html',records=attendance_records)
@app.route('/final_review')
def final_review():
    attendance_records = review_attendance()
    
   
    return render_template('final_review.html',records=attendance_records)

# Route to fetch all attendance data
@app.route('/get_attendance', methods=['GET'])
def get():
    at_list = get_attendance()
    print(at_list)
    return jsonify(at_list)
@app.route('/update_attendance', methods=['POST'])
def update():
    data = request.json  # Expecting { "id": 1, "marked": true }
    update_attendance(data)
    return jsonify({"message": "Attendance updated successfully"})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
