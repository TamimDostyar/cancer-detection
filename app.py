import os
import flask
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from run_model import run_model

app = Flask(__name__)
app.secret_key = 'secret-test-key'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            predicted_key, confidence = run_model(filepath)
            # Clean up the uploaded file
            os.remove(filepath)
            return render_template('result.html', 
                                 prediction=predicted_key, 
                                 confidence=confidence)
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('home'))
    
    flash('Invalid file type. Please upload a PNG, JPG, or JPEG file.')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

