import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from database.attendance import mark_attendance
import cv2

device = torch.device("cpu")

# Initialize models once
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load face embeddings
def load_model(load_path='face_recognition_model.pth'):
    loaded_data = torch.load(load_path, map_location=device)
    return loaded_data['face_embeddings'], loaded_data['names']

face_embeddings, known_face_names = load_model()
video_capture = cv2.VideoCapture(0)



def recognize_face(frame):
    """Detect and recognize faces in the frame."""
    img = Image.fromarray(frame)
    boxes, _ = mtcnn.detect(img)
    recognized_faces = []

    if boxes is not None:
        for box in boxes:
            face_img = img.crop(tuple(box))
            face_tensor = mtcnn(face_img, return_prob=False)

            if face_tensor is not None:
                with torch.no_grad():
                    emb = resnet(face_tensor.unsqueeze(0).to(device)).cpu()

                min_dist, identity = min(((torch.dist(emb, e).item(), n) for e, n in zip(face_embeddings, known_face_names)), default=(float('inf'), "Unknown"))
                mark_attendance(identity if min_dist < 0.8 else "Unknown")
                recognized_faces.append((identity if min_dist < 0.8 else "Unknown", tuple(map(int, box))))
    
    return recognized_faces


def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        recognized = recognize_face(frame_rgb)
        
        for name, (x1, y1, x2, y2) in recognized:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Release video resource when stopping
def stop_video():
    video_capture.release()
