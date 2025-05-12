import base64
import os
import json
import numpy as np
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch

# تهيئة MTCNN للكشف عن الوجوه
detector = MTCNN()

# تهيئة FaceNet لاستخراج الـ embeddings
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def decode_base64_image(image_data, filename):
    """
    فك base64 string وحفظه كملف صورة.
    """
    if not image_data or not isinstance(image_data, str):
        raise ValueError("Invalid or empty image data")
    if not image_data.startswith('data:image'):
        raise ValueError("Image data does not contain valid base64 string")
    
    try:
        image_data = image_data.split(',')[1]  # إزالة "data:image/jpeg;base64,"
        with open(filename, 'wb') as f:
            f.write(base64.b64decode(image_data))
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")

def get_face_embeddings(image_path):
    """
    استخراج face embeddings باستخدام MTCNN وFaceNet.
    """
    try:
        # فتح الصورة
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        # كشف الوجه بـ MTCNN
        faces = detector.detect_faces(image_np)
        if not faces:
            raise ValueError("No face detected in image")

        # استخراج أول وجه
        x, y, width, height = faces[0]['box']
        face = image_np[y:y+height, x:x+width]
        face_image = Image.fromarray(face).resize((160, 160))  # FaceNet يحتاج 160x160

        # تحويل الصورة إلى tensor
        face_tensor = torch.tensor(np.array(face_image)).permute(2, 0, 1).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0)  # إضافة batch dimension

        # استخراج الـ embedding
        with torch.no_grad():
            embedding = resnet(face_tensor).numpy().flatten()

        return embedding

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def compare_embeddings(embedding1, embedding2):
    """
    مقارنة embeddings باستخدام cosine similarity.
    """
    try:
        # تحويل إلى numpy arrays
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        # حساب cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        
        return similarity
    except Exception as e:
        print(f"Error comparing embeddings: {str(e)}")
        return 0.0

def save_user_data(user_id, user_name):
    """
    حفظ بيانات المستخدم في ملف JSON.
    """
    users = load_users()
    users.append({'id': user_id, 'name': user_name})
    
    try:
        with open('users.json', 'w') as f:
            json.dump(users, f)
    except Exception as e:
        raise Exception(f"Failed to save user data: {str(e)}")

def load_users():
    """
    تحميل بيانات المستخدمين من ملف JSON.
    """
    try:
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading users: {str(e)}")
        return []

def update_user(user_id, new_name):
    """
    تحديث اسم المستخدم.
    """
    users = load_users()
    for user in users:
        if user['id'] == user_id:
            user['name'] = new_name
            break
    try:
        with open('users.json', 'w') as f:
            json.dump(users, f)
    except Exception as e:
        raise Exception(f"Failed to update user: {str(e)}")

def delete_user(user_id):
    """
    حذف مستخدم.
    """
    users = load_users()
    users = [user for user in users if user['id'] != user_id]
    try:
        with open('users.json', 'w') as f:
            json.dump(users, f)
    except Exception as e:
        raise Exception(f"Failed to delete user: {str(e)}")