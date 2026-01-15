import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from config import Config
from chatbot import ChatbotInterface

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
chatbot = ChatbotInterface()

print('✅ Flask app initialized')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/caption', methods=['POST'])
def generate_caption():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '❌ No file'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': '❌ No file selected'}), 400
        
        if not Config.allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': '❌ Invalid file type'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        result = chatbot.process_image_upload(filepath, input_type='file')
        
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result), 200 if result['status'] == 'success' else 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'❌ {str(e)}'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({'status': 'success', 'stats': chatbot.get_stats()})

if __name__ == '__main__':
    print('🚀 Starting server...')
    print(f'🌐 http://localhost:{Config.PORT}')
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
