import os
import time
import subprocess
import threading
import webbrowser
from functools import wraps
from flask import Flask, request, jsonify, session, send_from_directory, redirect
from werkzeug.utils import secure_filename
from pipeline_v2 import predict_pipeline
from database import Database
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
app.secret_key = 'super_secret_teacher_key_123' # Change this in production

# Initialize Database
db = Database()

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max-limit for PDFs

# Session Configuration for Local Development
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True only if using HTTPS

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Login Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

# giriş sayfası
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing JSON body'}), 400
        
    username = data.get('username')
    password = data.get('password')
    
    # Verify with database
    user = db.verify_user(username, password)
    
    if user:
        session['logged_in'] = True
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['full_name'] = user['full_name'] or user['username']
        return jsonify({'success': True, 'user': {'username': user['username'], 'full_name': session['full_name']}})
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/user_info')
def user_info():
    if session.get('logged_in'):
        return jsonify({
            'logged_in': True,
            'user': {
                'id': session['user_id'],
                'username': session['username'],
                'full_name': session['full_name']
            }
        })
    return jsonify({'logged_in': False}), 401

@app.route('/logout')
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/')
def index():
    # Redirect to the Frontend Application
    return redirect("http://localhost:5173")

# profil sayfası
@app.route('/profile')
@login_required
def profile():
    user_id = session.get('user_id')
    user = db.get_user_by_id(user_id)
    predictions = db.get_user_predictions(user_id, limit=10)
    
    # Calculate stats
    total_predictions = len(db.get_user_predictions(user_id, limit=1000))
    
    return jsonify({
        'user': user,
        'predictions': predictions,
        'total_predictions': total_predictions
    })

@app.route('/edit_profile', methods=['POST'])
@login_required
def edit_profile():
    user_id = session.get('user_id')
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing data'}), 400
        
    full_name = data.get('full_name')
    email = data.get('email')
    
    if db.update_user_info(user_id, full_name, email):
        session['full_name'] = full_name
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Failed to update profile'}), 500

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    user_id = session.get('user_id')
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing data'}), 400

    current_password = data.get('current_password')
    new_password = data.get('new_password')
    confirm_password = data.get('confirm_password')
    
    if new_password != confirm_password:
        return jsonify({'error': 'New passwords do not match!'}), 400
    
    if db.change_password(user_id, current_password, new_password):
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Current password is incorrect!'}), 400

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid cache/overwrite issues
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check if PDF
        if filename.lower().endswith('.pdf'):
            return process_pdf(filepath, filename)
        else:
            return process_image(filepath, filename)
            
    return jsonify({'error': 'Invalid file type'}), 400

def process_image(filepath, filename):
    """Process a single image"""
    try:
        result = predict_pipeline(filepath)
        
        if result:
            # Save to database
            user_id = session.get('user_id')
            if user_id:
                db.save_prediction(
                    user_id=user_id,
                    image_name=filename,
                    raw_text=result['raw_text'],
                    corrected_text=result['text']
                )
            
            # Ensure URL is accessible and points to the VISUALIZED image
            image_url = result['image_url']
            if not image_url.startswith('/'):
                image_url = '/' + image_url
            
            return jsonify({
                'success': True,
                'image_url': image_url,
                'text': result['text'],
                'raw_text': result['raw_text']
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

def process_pdf(filepath, filename):
    """Process PDF file using PyMuPDF - convert pages to images and process each"""
    try:
        # Open PDF with PyMuPDF
        print(f"Converting PDF to images: {filepath}")
        pdf_document = fitz.open(filepath)
        total_pages = len(pdf_document)  # Store page count before closing
        
        all_results = []
        combined_text = []
        combined_raw_text = []
        
        # Process each page
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            
            # Convert page to image (high resolution)
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Save page as temporary image
            page_filename = f"temp_page_{page_num + 1}_{filename.replace('.pdf', '.png')}"
            temp_image_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                page_filename
            )
            img.save(temp_image_path, 'PNG')
            
            # Process the page
            result = predict_pipeline(temp_image_path)
            
            if result:
                # Use the VISUALIZED image from pipeline
                vis_url = result['image_url']
                if not vis_url.startswith('/'):
                    vis_url = '/' + vis_url
                    
                all_results.append({
                    'page': page_num + 1,
                    'text': result['text'],
                    'raw_text': result['raw_text'],
                    'image_url': vis_url
                })
                combined_text.append(f"[Page {page_num + 1}] {result['text']}")
                combined_raw_text.append(f"[Page {page_num + 1}] {result['raw_text']}")
        
        pdf_document.close()
        
        # Save to database
        user_id = session.get('user_id')
        if user_id:
            db.save_prediction(
                user_id=user_id,
                image_name=filename,
                raw_text="\n".join(combined_raw_text),
                corrected_text="\n".join(combined_text)
            )
        
        # Create PDF URL
        pdf_path = create_annotated_pdf(all_results, filename)
        pdf_url = f"/static/uploads/{os.path.basename(pdf_path)}" if pdf_path else None
        
        return jsonify({
            'success': True,
            'is_pdf': True,
            'pages': all_results,
            'text': "\n\n".join(combined_text),
            'raw_text': "\n\n".join(combined_raw_text),
            'total_pages': total_pages,
            'pdf_url': pdf_url
        })
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'PDF processing failed: {str(e)}'}), 500

def create_annotated_pdf(pages_data, original_filename):
    """Creates a PDF from annotated images"""
    try:
        # Create output filename
        name = os.path.splitext(original_filename)[0]
        output_filename = f"result_web_{name}.pdf"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        doc = fitz.open()
        
        for page_data in pages_data:
            # Extract filename from URL 
            img_fname = os.path.basename(page_data['image_url'])
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_fname)
            
            if os.path.exists(img_path):
                # Open image
                img_doc = fitz.open(img_path)
                pdfbytes = img_doc.convert_to_pdf()
                img_doc.close()
                
                imgPDF = fitz.open("pdf", pdfbytes)
                page = doc.new_page(width = imgPDF[0].rect.width, height = imgPDF[0].rect.height)
                page.show_pdf_page(page.rect, imgPDF, 0)
                imgPDF.close()
            else:
                print(f"Warning: Image not found for PDF creation: {img_path}")
                
        doc.save(output_path)
        doc.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return None

if __name__ == '__main__':
    # Only run the startup script in the main process, not the reloader
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        print("\n" + "="*60)
        print("ProOCR - Handwriting Recognition API Backend")
        print("="*60)
        print("Database initialized")
        print("Models loaded")
        print("Server starting...")

        # Start Frontend Automatically
        print("Starting React Frontend...")
        frontend_dir = os.path.join(os.getcwd(), 'frontend')
        npm_cmd = 'npm.cmd' if os.name == 'nt' else 'npm'

        try:
            # Run npm run dev in background
            subprocess.Popen([npm_cmd, 'run', 'dev'], cwd=frontend_dir, shell=True)
            print("Frontend started!")

            # Open Browser automatically after a short delay
            print("Opening browser at http://localhost:5173...")
            def open_browser():
                time.sleep(3) # Wait for servers to spin up
                webbrowser.open("http://localhost:5173")

            threading.Thread(target=open_browser).start()

        except Exception as e:
            print(f"Failed to start frontend automatically: {e}")
            print("   Please run 'cd frontend && npm run dev' manually.")

        print("\nAPI Listener:")
        print("   http://127.0.0.1:5000 (Backend)")
        print("\nPress CTRL+C to stop the server")
        print("="*60 + "\n")
    
    app.run(debug=True, port=5000)
