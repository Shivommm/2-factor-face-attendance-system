import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define device (CPU in this case)
device = torch.device('cpu')

# Define MTCNN for face detection
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Define InceptionResnetV1 for face embedding (lightweight model)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define dataset directory
data_dir = './dataset'

# Check if the dataset directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

# Check if the dataset directory contains subfolders
subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
if not subfolders:
    raise FileNotFoundError(f"No subfolders found in dataset directory: {data_dir}")

# Check if the subfolders contain valid image files
for subfolder in subfolders:
    subfolder_path = os.path.join(data_dir, subfolder)
    image_files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'))]
    if not image_files:
        print(f"Warning: No valid image files found in subfolder: {subfolder_path}")

# Load dataset
dataset = datasets.ImageFolder(data_dir)
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=lambda x: x[0])

# Create lists for storing face embeddings and corresponding names
face_embeddings = []
names = []

# Iterate through the dataset and extract face embeddings
for img, idx in tqdm(loader):
    face, prob = mtcnn(img, return_prob=True)
    if face is not None and prob > 0.90:
        emb = resnet(face.unsqueeze(0).to(device)).detach().cpu()
        face_embeddings.append(emb)
        names.append(dataset.idx_to_class[idx])

# ... (rest of your code - including the changes from previous response)

# Function to save the trained model (face embeddings and names)
def save_model(face_embeddings, names, save_path='face_recognition_model.pth'):
    torch.save({
        'face_embeddings': face_embeddings,
        'names': names
    }, save_path)
    print(f"Model saved to {save_path}")


# After training, save the model:
save_model(face_embeddings, names)
