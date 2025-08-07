from flask import Flask, render_template, request, redirect, url_for, session, flash
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage # Import storage
import requests
import os
import random
import json
import re
import time
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- App and Firebase Initialization ---
app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    cred = credentials.Certificate("accountKey.json")
    if not firebase_admin._apps:
        # Add your storageBucket ID here
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'nisarga-773f0.firebasestorage.app' 
        })
    db = firestore.client()
    bucket = storage.bucket() # Initialize the storage bucket
except Exception as e:
    logger.critical(f"FATAL: Failed to initialize Firebase: {e}")

# --- API Configuration ---
OPENROUTER_API_KEY = os.environ.get("#########")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# --- New Function to Save to Firebase Storage ---
def save_results_to_storage(result_data):
    """Saves the detailed quiz results JSON to Firebase Storage."""
    try:
        student_email = session.get('email')
        subject = result_data.get('subject', 'general_topic')

        if not student_email:
            logger.error("Cannot save to storage: user email not found in session.")
            return

        # Sanitize names for use in file paths
        sanitized_email = re.sub(r'[.@]', '_', student_email)
        sanitized_subject = re.sub(r'[^a-zA-Z0-9]', '_', subject)
        
        # Create a unique filename with a timestamp
        timestamp = int(time.time())
        file_name = f"{timestamp}_{sanitized_subject}.json"
        
        # Define the full, organized path in Firebase Storage
        storage_path = f"quiz_results/students/{sanitized_email}/{file_name}"
        
        # Convert the results dictionary to a JSON string (use default=str to handle Timestamps)
        result_json = json.dumps(result_data, indent=4, default=str)

        # Get a reference to the blob (file) in storage and upload
        blob = bucket.blob(storage_path)
        blob.upload_from_string(result_json, content_type='application/json')
        
        logger.info(f"Successfully uploaded results to Firebase Storage at: {storage_path}")
        flash("✅ Your result has been saved to Storage!", 'success')

    except Exception as e:
        logger.error(f"Error saving results to Firebase Storage: {e}")
        flash("Error saving results to Storage.", 'error')


# --- ML Model Function ---
def predict_pass_fail(db, student_uid, current_accuracy, current_avg_time):
    try:
        # This function can be enhanced to query Storage instead of Firestore if needed
        docs = db.collection('quiz_results').where('student_uid', '==', student_uid).stream()
        historical_data = [doc.to_dict() for doc in docs]
        if len(historical_data) < 3:
            X_train = np.array([[0.5, 10], [0.7, 8], [0.3, 15]])
            y_train = np.array([0, 1, 0])
        else:
            X_train = np.array([[d['accuracy'], d['average_time']] for d in historical_data])
            y_train = np.array([1 if d['accuracy'] >= 0.6 else 0 for d in historical_data])
        model = LogisticRegression()
        model.fit(X_train, y_train)
        prediction = model.predict([[current_accuracy, current_avg_time]])
        return 'Pass' if prediction[0] == 1 else 'Fail'
    except Exception as e:
        logger.error(f"Error in pass/fail prediction: {e}")
        return "Prediction Error"

# --- Helper Function for Parsing ---
def parse_questions(text):
    questions = []
    start_index = text.find("Question:")
    if start_index == -1: return []
    relevant_text = text[start_index:]
    question_blocks = re.split(r'(?=Question:)', relevant_text.strip())
    for block in question_blocks:
        if not block.strip(): continue
        lines = block.strip().split('\n')
        try:
            q_data = {}
            options = {}
            for line in lines:
                clean_line = line.strip()
                if clean_line.lower().startswith('question:'):
                    q_data['question'] = clean_line.split(':', 1)[1].strip()
                elif re.match(r'^[A-D]:', clean_line, re.IGNORECASE):
                    key, value = clean_line.split(':', 1)
                    options[key.strip().upper()] = value.strip()
                elif clean_line.lower().startswith('answer:'):
                    q_data['answer'] = clean_line.split(':', 1)[1].strip().upper().replace('.', '')
                elif clean_line.lower().startswith('explanation:'):
                    q_data['explanation'] = clean_line.split(':', 1)[1].strip()
            if all(k in q_data for k in ['question', 'answer', 'explanation']) and len(options) == 4:
                q_data['options'] = options
                questions.append(q_data)
        except Exception as e:
            logger.error(f"Error parsing question block: {e}")
    return questions

# --- Routes ---
@app.route('/')
def index():
    if 'uid' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.create_user(email=email, password=password, display_name=name)
            db.collection('users').document(user.uid).set({'name': name, 'email': email})
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Error creating account: {e}", 'error')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        try:
            user = auth.get_user_by_email(email)
            session['uid'] = user.uid
            session['email'] = user.email
            session['name'] = user.display_name
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash("Login failed. Check your credentials.", 'error')
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'uid' not in session:
        return redirect(url_for('login'))
    if request.args.get('action') == 'play_again':
        session.pop('quiz_results', None)
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        subject = request.form['subject']
        difficulty = request.form.get('difficulty', 'Intermediate')
        try:
            total_q = int(request.form['total_questions'])
        except ValueError:
            flash("Please enter a valid number for questions.", 'error')
            return redirect(url_for('dashboard'))
        prompt = (
            f"Generate {total_q} {difficulty} level multiple choice quiz questions on the topic: {subject}. "
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Your response MUST begin directly with 'Question:'. Do not add any introductory text like 'Sure, here are...' or any other conversational fillers.\n"
            "2. Your entire response must consist ONLY of the question blocks. Do not add a summary or concluding text at the end.\n"
            "3. Each question block must be separated by exactly one blank line.\n"
            "4. Follow this format precisely for each block:\n\n"
            "Question: [The question text]\n"
            "A: [Option A]\n"
            "B: [Option B]\n"
            "C: [Option C]\n"
            "D: [Option D]\n"
            "Answer: [The correct option letter, e.g., B]\n"
            "Explanation: [A brief explanation for why the answer is correct]"
        )
        try:
            headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
            payload = {"model": "meta-llama/llama-3-8b-instruct", "messages": [{"role": "user", "content": prompt}]}
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            questions_text = response.json()['choices'][0]['message']['content']
            questions = parse_questions(questions_text)
            if not questions:
                flash("Failed to parse questions from AI. Please try again.", 'error')
                return redirect(url_for('dashboard'))
            session['questions'] = questions
            session['current'] = 0
            session['subject'] = subject
            session['difficulty'] = difficulty
            session.pop('quiz_results', None)
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
            return redirect(url_for('dashboard'))
    quiz_results = session.get('quiz_results')
    if quiz_results:
        return render_template('dashboard.html', name=session.get('name'), results=quiz_results)
    current_q_index = session.get('current')
    all_questions = session.get('questions')
    if all_questions and current_q_index is not None and current_q_index < len(all_questions):
        current_q = all_questions[current_q_index]
        return render_template('dashboard.html', name=session.get('name'), question=current_q, questions=all_questions)
    return render_template('dashboard.html', name=session.get('name'))

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    if 'uid' not in session:
        return redirect(url_for('login'))

    answer = request.form.get('answer')
    time_taken = int(float(request.form.get('time_taken', '0')))
    current_idx = session.get('current', 0)
    questions = session.get('questions', [])
    
    if current_idx < len(questions):
        questions[current_idx]['submitted'] = answer
        questions[current_idx]['time_taken'] = time_taken
        session['questions'] = questions
    
    session['current'] = current_idx + 1

    if session['current'] >= len(questions):
        correct_count = sum(1 for q in questions if q.get('submitted', '').upper() == q.get('answer', '').upper())
        total_questions = len(questions)
        accuracy = correct_count / total_questions if total_questions > 0 else 0
        average_time = sum(q.get('time_taken', 0) for q in questions) / total_questions if total_questions > 0 else 0
        student_uid = session.get('uid')

        pass_fail_prediction = predict_pass_fail(db, student_uid, accuracy, average_time)

        result_data = {
            'student_name': session.get('name') or 'Unknown Student',
            'student_uid': student_uid,
            'subject': session.get('subject', 'General'),
            'difficulty': session.get('difficulty', 'Intermediate'),
            'score': correct_count,
            'total_questions': total_questions,
            'details': questions,
            'accuracy': accuracy,
            'average_time': average_time,
            'pass_fail_prediction': pass_fail_prediction,
        }
        
        # --- LOGIC CHANGED: Call the function to save to Storage ---
        save_results_to_storage(result_data)
        
        session['quiz_results'] = result_data
        
        session.pop('questions', None)
        session.pop('current', None)
        session.pop('subject', None)
        session.pop('difficulty', None)
    
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)

