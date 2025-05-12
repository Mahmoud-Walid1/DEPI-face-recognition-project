from flask import Flask, request, jsonify, render_template
import os
import json
import datetime
from collections import defaultdict
import glob
import numpy as np
from utils import decode_base64_image, get_face_embeddings, save_user_data, load_users, update_user, delete_user, compare_embeddings

app = Flask(__name__)

# قاموس مؤقت لحفظ الـ embeddings
known_embeddings = {}

# دالة لحفظ embeddings إلى ملف محلي
def save_embeddings():
    try:
        embeddings_to_save = {user_id: [emb.tolist() for emb in embeddings] for user_id, embeddings in known_embeddings.items()}
        with open('embeddings.json', 'w') as f:
            json.dump(embeddings_to_save, f, indent=2)
        print("Embeddings saved to embeddings.json locally")
    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")

# دالة لتحميل embeddings من ملف محلي
def load_embeddings():
    global known_embeddings
    try:
        if os.path.exists('embeddings.json'):
            with open('embeddings.json', 'r') as f:
                embeddings_data = json.load(f)
            known_embeddings = {}
            for user_id, embeddings in embeddings_data.items():
                try:
                    known_embeddings[user_id] = [np.array(emb) for emb in embeddings]
                    print(f"Loaded embeddings for user {user_id}: {len(embeddings)} vectors")
                except ValueError as ve:
                    print(f"Error converting embeddings for user {user_id} to numpy array: {str(ve)}")
            print(f"Total users loaded: {len(known_embeddings)}")
        else:
            print("embeddings.json does not exist, starting with empty embeddings")
    except Exception as e:
        print(f"Error loading embeddings file: {str(e)}")

# دالة لحفظ سجل حضور
def save_attendance(user_id, status):
    records = load_attendance()
    record = {
        'ID': user_id,
        'Name': get_user_name(user_id),
        'Date': datetime.datetime.now().strftime('%Y-%m-%d'),
        'Time': datetime.datetime.now().strftime('%H:%M:%S'),
        'Status': status
    }
    records.append(record)
    try:
        with open('attendance.json', 'w') as f:
            json.dump(records, f, indent=2)
        print(f"Attendance record saved for user {user_id}: {status}")
    except Exception as e:
        print(f"Error saving attendance: {str(e)}")

# دالة لتحميل سجلات الحضور
def load_attendance():
    try:
        if os.path.exists('attendance.json'):
            with open('attendance.json', 'r') as f:
                return json.load(f)
        print("attendance.json does not exist, creating empty file")
        with open('attendance.json', 'w') as f:
            json.dump([], f)
        return []
    except Exception as e:
        print(f"Error loading attendance: {str(e)}")
        return []

# دالة لحفظ محاولة تعرف
def save_recognition_attempt(user_id, similarity, result, error=None):
    attempts = load_recognition_attempts()
    attempt = {
        'ID': user_id or 'Unknown',
        'Similarity': float(similarity) if similarity is not None else None,
        'Result': result,
        'Error': error,
        'Date': datetime.datetime.now().strftime('%Y-%m-%d'),
        'Time': datetime.datetime.now().strftime('%H:%M:%S')
    }
    attempts.append(attempt)
    try:
        with open('recognition_attempts.json', 'w') as f:
            json.dump(attempts, f, indent=2)
        print(f"Recognition attempt saved: {attempt}")
    except Exception as e:
        print(f"Error saving recognition attempt: {str(e)}")

# دالة لتحميل محاولات التعرف
def load_recognition_attempts():
    try:
        if os.path.exists('recognition_attempts.json'):
            with open('recognition_attempts.json', 'r') as f:
                return json.load(f)
        print("recognition_attempts.json does not exist, creating empty file")
        with open('recognition_attempts.json', 'w') as f:
            json.dump([], f)
        return []
    except Exception as e:
        print(f"Error loading recognition attempts: {str(e)}")
        return []

# دالة لجلب اسم المستخدم بناءً على ID
def get_user_name(user_id):
    users = load_users()
    for user in users:
        if user['id'] == user_id:
            return user['name']
    return 'Unknown'

# دالة لحساب تقارير الأداء
def get_performance():
    try:
        records = load_attendance()
        attempts = load_recognition_attempts()
        performance = defaultdict(lambda: {
            'check_ins': 0,
            'check_outs': 0,
            'days': set(),
            'total_attempts': 0,
            'successful_attempts': 0,
            'false_accepts': 0,
            'errors': 0
        })
        
        for record in records:
            user_id = record['ID']
            status = record['Status']
            date = record['Date']
            
            if status == 'Check-in':
                performance[user_id]['check_ins'] += 1
                performance[user_id]['days'].add(date)
            elif status == 'Check-out':
                performance[user_id]['check_outs'] += 1
                performance[user_id]['days'].add(date)
        
        for attempt in attempts:
            user_id = attempt['ID']
            result = attempt['Result']
            
            performance[user_id]['total_attempts'] += 1
            if result == 'Success':
                performance[user_id]['successful_attempts'] += 1
            elif result == 'FalseAccept':
                performance[user_id]['false_accepts'] += 1
            elif result in ['NoMatch', 'NoFace']:
                performance[user_id]['errors'] += 1
        
        result = []
        for user_id, stats in performance.items():
            total_attempts = stats['total_attempts']
            successful_attempts = stats['successful_attempts']
            false_accepts = stats['false_accepts']
            
            accuracy = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0.0
            total_accepts = successful_attempts + false_accepts
            far = (false_accepts / total_accepts * 100) if total_accepts > 0 else 0.0
            
            result.append({
                'ID': user_id,
                'Name': get_user_name(user_id),
                'TotalCheckIns': stats['check_ins'],
                'TotalCheckOuts': stats['check_outs'],
                'UniqueDays': len(stats['days']),
                'Accuracy': round(accuracy, 2),
                'FAR': round(far, 2),
                'RecognitionErrors': stats['errors']
            })
        
        print(f"Performance report generated: {result}")
        return result
    except Exception as e:
        print(f"Error generating performance report: {str(e)}")
        return []

# دالة لتوليد user_id فريد
def generate_unique_user_id(user_name, existing_ids):
    base_id = user_name.replace(" ", "_").lower()
    counter = 1
    user_id = base_id
    while user_id in existing_ids:
        user_id = f"{base_id}_{counter}"
        counter += 1
    return user_id

# دالة لحفظ بيانات المستخدم في ملف محلي
def save_user_data(user_id, user_name):
    users = load_users()
    users = [user for user in users if user['id'] != user_id]  # إزالة المستخدم القديم إذا موجود
    users.append({'id': user_id, 'name': user_name})
    try:
        with open('users.json', 'w') as f:
            json.dump(users, f, indent=2)
        print(f"User data saved to users.json for user {user_id}")
    except Exception as e:
        print(f"Error saving user data: {str(e)}")

# دالة لتحميل بيانات المستخدمين من ملف محلي
def load_users():
    try:
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                return json.load(f)
        print("users.json does not exist, creating empty file")
        with open('users.json', 'w') as f:
            json.dump([], f)
        return []
    except Exception as e:
        print(f"Error loading users: {str(e)}")
        return []

# تحميل الـ embeddings عند بدء السيرفر
load_embeddings()

@app.route('/')
def index():
    return render_template('index.html', message=None, message_type=None)

@app.route('/register')
def register():
    return render_template('register.html', message=None, message_type=None)

@app.route('/attendance')
def attendance():
    return render_template('attendance.html', message=None, message_type=None)

@app.route('/performance_report')
def performance_report():
    return render_template('performance_report.html', message=None, message_type=None)

@app.route('/capture', methods=['POST'])
def capture():
    user_id = request.form.get('user_id')
    user_name = request.form.get('user_name')
    image_data_1 = request.form.get('image_data_1')
    image_data_2 = request.form.get('image_data_2')

    print(f"Received data: user_id={user_id}, user_name={user_name}, "
          f"image_data_1={'set' if image_data_1 else 'None'}, "
          f"image_data_2={'set' if image_data_2 else 'None'}")

    if not all([user_id, user_name, image_data_1, image_data_2]):
        missing = []
        if not user_id:
            missing.append('user_id')
        if not user_name:
            missing.append('user_name')
        if not image_data_1:
            missing.append('image_data_1')
        if not image_data_2:
            missing.append('image_data_2')
        print(f"Missing fields: {missing}")
        return render_template('register.html', message=f'Missing fields: {", ".join(missing)}', message_type='error')

    try:
        os.makedirs('temp', exist_ok=True)
        image_path_1 = 'temp/image1.jpg'
        image_path_2 = 'temp/image2.jpg'
        decode_base64_image(image_data_1, image_path_1)
        decode_base64_image(image_data_2, image_path_2)

        embedding1 = get_face_embeddings(image_path_1)
        embedding2 = get_face_embeddings(image_path_2)

        if embedding1 is None or embedding2 is None:
            os.remove(image_path_1)
            os.remove(image_path_2)
            print("No face detected in one or both images")
            return render_template('register.html', message='No face detected in one or both images', message_type='error')

        known_embeddings[user_id] = [embedding1, embedding2]
        save_user_data(user_id, user_name)
        save_embeddings()

        os.remove(image_path_1)
        os.remove(image_path_2)

        print(f"User {user_id} registered successfully with embeddings: {len(known_embeddings[user_id])}")
        return render_template('register.html', message='User registered successfully', message_type='success')

    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
        if os.path.exists(image_path_1):
            os.remove(image_path_1)
        if os.path.exists(image_path_2):
            os.remove(image_path_2)
        return render_template('register.html', message=str(ve), message_type='error')
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        if os.path.exists(image_path_1):
            os.remove(image_path_1)
        if os.path.exists(image_path_2):
            os.remove(image_path_2)
        return render_template('register.html', message=f'Failed to process images: {str(e)}', message_type='error')

@app.route('/bulk_register', methods=['POST'])
def bulk_register():
    try:
        global known_embeddings
        known_embeddings = {}  # إعادة تعيين الـ embeddings لتسجيل جديد
        with open('users.json', 'w') as f:
            json.dump([], f)  # إعادة تعيين users.json

        user_folders = [d for d in glob.glob('users_photos/*') if os.path.isdir(d)]
        registered_users = []
        errors = []

        print(f"Found {len(user_folders)} folders in users_photos")
        for user_folder in user_folders:
            user_name = os.path.basename(user_folder)
            user_id = generate_unique_user_id(user_name, {user['id'] for user in load_users()})
            print(f"Generated user_id: {user_id} for user_name: {user_name}")

            image_paths = glob.glob(os.path.join(user_folder, '*.jpg'))[:2]  # أول صورتين
            print(f"Found {len(image_paths)} images for {user_name}: {image_paths}")
            if not image_paths:
                errors.append(f"No images found for user {user_name}")
                continue

            embeddings = []
            for image_path in image_paths:
                print(f"Processing image: {image_path}")
                try:
                    embedding = get_face_embeddings(image_path)
                    if embedding is None:
                        print(f"No face detected in {image_path}")
                        errors.append(f"No face detected in image {image_path} for user {user_name}")
                        continue
                    embeddings.append(embedding)
                    print(f"Generated embedding for {image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    errors.append(f"Error processing image {image_path} for user {user_name}: {str(e)}")
                    continue

            if not embeddings:
                print(f"No valid embeddings for {user_name}")
                errors.append(f"No valid embeddings generated for user {user_name}")
                continue

            known_embeddings[user_id] = embeddings
            save_user_data(user_id, user_name)
            save_embeddings()
            registered_users.append({'user_id': user_id, 'user_name': user_name})
            print(f"Successfully registered user {user_id} ({user_name}) with {len(embeddings)} embeddings")

        print(f"Registration complete. Registered: {len(registered_users)}, Errors: {len(errors)}")
        if not registered_users and errors:
            return jsonify({'status': 'error', 'message': 'No users registered', 'errors': errors}), 400

        return jsonify({
            'status': 'success',
            'message': f'Registered {len(registered_users)} users',
            'registered_users': registered_users,
            'errors': errors
        })

    except Exception as e:
        print(f"Unexpected error in bulk_register: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Failed to process bulk registration: {str(e)}'}), 500

@app.route('/recognize', methods=['POST'])
def recognize():
    image_data = request.form.get('image_data')
    print(f"Received recognition request: image_data={'set' if image_data else 'None'}")

    if not image_data:
        print("Missing image_data")
        save_recognition_attempt(None, None, 'NoFace', error='No image provided')
        return render_template('attendance.html', message='No image provided', message_type='error')

    try:
        image_path = 'temp/unknown.jpg'
        decode_base64_image(image_data, image_path)
        unknown_embedding = get_face_embeddings(image_path)

        if unknown_embedding is None:
            os.remove(image_path)
            print("No face detected in image")
            save_recognition_attempt(None, None, 'NoFace', error='No face detected')
            return render_template('attendance.html', message='No face detected in image', message_type='error')

        max_similarity = 0.0
        matched_user_id = None
        for user_id, embeddings in known_embeddings.items():
            for known_embedding in embeddings:
                similarity = compare_embeddings(unknown_embedding, known_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    matched_user_id = user_id

        if max_similarity > 0.6:
            os.remove(image_path)
            print(f"User {matched_user_id} recognized with similarity {max_similarity}")
            save_attendance(matched_user_id, 'Check-in')
            save_recognition_attempt(matched_user_id, max_similarity, 'Success')
            return render_template('attendance.html', message=f'User {get_user_name(matched_user_id)} checked in', message_type='success')
        else:
            os.remove(image_path)
            print("No match found")
            save_recognition_attempt(None, max_similarity, 'NoMatch')
            return render_template('attendance.html', message='No match found', message_type='error')

    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
        if os.path.exists(image_path):
            os.remove(image_path)
        save_recognition_attempt(None, None, 'NoFace', error=str(ve))
        return render_template('attendance.html', message=str(ve), message_type='error')
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        if os.path.exists(image_path):
            os.remove(image_path)
        save_recognition_attempt(None, None, 'NoFace', error=str(e))
        return render_template('attendance.html', message=f'Failed to process images: {str(e)}', message_type='error')

@app.route('/checkout', methods=['POST'])
def checkout():
    image_data = request.form.get('image_data')
    print(f"Received checkout request: image_data={'set' if image_data else 'None'}")

    if not image_data:
        print("Missing image_data")
        save_recognition_attempt(None, None, 'NoFace', error='No image provided')
        return render_template('attendance.html', message='No image provided', message_type='error')

    try:
        image_path = 'temp/unknown.jpg'
        decode_base64_image(image_data, image_path)
        unknown_embedding = get_face_embeddings(image_path)

        if unknown_embedding is None:
            os.remove(image_path)
            print("No face detected in image")
            save_recognition_attempt(None, None, 'NoFace', error='No face detected')
            return render_template('attendance.html', message='No face detected in image', message_type='error')

        max_similarity = 0.0
        matched_user_id = None
        for user_id, embeddings in known_embeddings.items():
            for known_embedding in embeddings:
                similarity = compare_embeddings(unknown_embedding, known_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    matched_user_id = user_id

        if max_similarity > 0.6:
            os.remove(image_path)
            print(f"User {matched_user_id} checked out with similarity {max_similarity}")
            save_attendance(matched_user_id, 'Check-out')
            save_recognition_attempt(matched_user_id, max_similarity, 'Success')
            return render_template('attendance.html', message=f'User {get_user_name(matched_user_id)} checked out', message_type='success')
        else:
            os.remove(image_path)
            print("No match found")
            save_recognition_attempt(None, max_similarity, 'NoMatch')
            return render_template('attendance.html', message='No match found', message_type='error')

    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
        if os.path.exists(image_path):
            os.remove(image_path)
        save_recognition_attempt(None, None, 'NoFace', error=str(ve))
        return render_template('attendance.html', message=str(ve), message_type='error')
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        if os.path.exists(image_path):
            os.remove(image_path)
        save_recognition_attempt(None, None, 'NoFace', error=str(e))
        return render_template('attendance.html', message=f'Failed to process images: {str(e)}', message_type='error')

@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    try:
        records = load_attendance()
        print(f"Returning attendance records: {records}")
        return jsonify(records)
    except Exception as e:
        print(f"Error in /get_attendance: {str(e)}")
        return jsonify({'error': 'Failed to load attendance records'}), 500

@app.route('/get_performance', methods=['GET'])
def get_performance_route():
    try:
        performance = get_performance()
        print(f"Returning performance report: {performance}")
        return jsonify(performance)
    except Exception as e:
        print(f"Error in /get_performance: {str(e)}")
        return jsonify({'error': 'Failed to load performance report'}), 500

@app.route('/users', methods=['GET'])
def get_users():
    users = load_users()
    print(f"Fetched users: {users}")
    return jsonify(users)

@app.route('/update_user', methods=['POST'])
def update_user_route():
    data = request.get_json()
    user_id = data.get('id')
    new_name = data.get('name')

    if not user_id or not new_name:
        print("Missing user_id or name in update request")
        return jsonify({'status': 'error', 'message': 'Missing user_id or name'})

    try:
        update_user(user_id, new_name)
        print(f"User {user_id} updated to name {new_name}")
        return jsonify({'status': 'success', 'message': 'User updated successfully'})
    except Exception as e:
        print(f"Error updating user: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/delete_user', methods=['POST'])
def delete_user_route():
    data = request.get_json()
    user_id = data.get('id')

    if not user_id:
        print("Missing user_id in delete request")
        return jsonify({'status': 'error', 'message': 'Missing user_id'})

    try:
        delete_user(user_id)
        if user_id in known_embeddings:
            del known_embeddings[user_id]
            save_embeddings()
        print(f"User {user_id} deleted")
        return jsonify({'status': 'success', 'message': 'User deleted successfully'})
    except Exception as e:
        print(f"Error deleting user: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)