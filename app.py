import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from config import Config
from chatbot import create_chatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)  # Enable CORS for frontend integration

# Create upload folder if it doesn't exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Initialize chatbot
chatbot = create_chatbot()

logger.info('✅ Flask application initialized')
logger.info(f'✅ Upload folder: {Config.UPLOAD_FOLDER}')
logger.info(f'✅ Max file size: {Config.MAX_CONTENT_LENGTH / (1024*1024):.0f} MB')


@app.route('/')
def index():
    """
    Home page
    Simple: Ye main page hai jahan users images upload kar sakte hain
    Hyderabadi: Ye main page hai jahan users images upload kar sakte hain
    """
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring
    
    Returns:
        JSON response with service status
    """
    stats = chatbot.get_stats()
    return jsonify({
        'status': 'healthy',
        'service': 'Image Caption Generator',
        'model': Config.HF_MODEL_NAME,
        'stats': stats
    })


@app.route('/api/caption', methods=['POST'])
def generate_caption():
    """
    Main endpoint for caption generation
    
    Accepts:
        - Multipart file upload (image file)
        - JSON with base64 encoded image
        - JSON with image URL
    
    Returns:
        JSON: {
            'status': str,
            'caption': str,
            'message': str,
            'timestamp': str
        }
    
    Simple: Ye endpoint images accept karta hai aur captions generate karta hai
    Hyderabadi: Ye endpoint images lete hai aur captions banate hai
    """
    try:
        # Case 1: File upload
        if 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': '❌ No file selected'
                }), 400
            
            if not Config.allowed_file(file.filename):
                return jsonify({
                    'status': 'error',
                    'message': f'❌ Invalid file type. Allowed: {", ".join(Config.ALLOWED_EXTENSIONS)}'
                }), 400
            
            # Save file securely
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            logger.info(f'📁 File saved: {filepath}')
            
            # Generate caption
            result = chatbot.process_image_upload(filepath, input_type='file')
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
                logger.info(f'🗑️ Cleaned up: {filepath}')
            except:
                pass
            
            return jsonify(result), 200 if result['status'] == 'success' else 500
        
        # Case 2: JSON with URL or base64
        elif request.is_json:
            data = request.get_json()
            
            # URL-based caption
            if 'url' in data:
                url = data['url']
                logger.info(f'🌐 Processing URL: {url}')
                result = chatbot.process_image_upload(url, input_type='url')
                return jsonify(result), 200 if result['status'] == 'success' else 500
            
            # Base64-based caption (for frontend integration)
            elif 'image' in data:
                import base64
                from io import BytesIO
                
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(data['image'])
                    image = Image.open(BytesIO(image_data))
                    
                    logger.info('🖼️ Processing base64 image')
                    result = chatbot.process_image_upload(image, input_type='pil')
                    return jsonify(result), 200 if result['status'] == 'success' else 500
                    
                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': f'❌ Invalid base64 image: {str(e)}'
                    }), 400
            
            else:
                return jsonify({
                    'status': 'error',
                    'message': '❌ Missing "url" or "image" in request'
                }), 400
        
        else:
            return jsonify({
                'status': 'error',
                'message': '❌ Invalid request format'
            }), 400
    
    except Exception as e:
        logger.error(f'❌ API error: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': f'❌ Server error: {str(e)}'
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """
    Get conversation history
    
    Query params:
        - limit: Maximum number of entries (default: 20)
    
    Returns:
        JSON list of conversation entries
    """
    limit = request.args.get('limit', 20, type=int)
    history = chatbot.get_conversation_history(limit=limit)
    
    return jsonify({
        'status': 'success',
        'count': len(history),
        'history': history
    })


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    result = chatbot.clear_history()
    return jsonify(result)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get chatbot statistics"""
    stats = chatbot.get_stats()
    return jsonify({
        'status': 'success',
        'stats': stats
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'status': 'error',
        'message': f'❌ File too large. Maximum size: {Config.MAX_CONTENT_LENGTH / (1024*1024):.0f} MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': '❌ Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 errors"""
    logger.error(f'❌ Internal server error: {str(error)}')
    return jsonify({
        'status': 'error',
        'message': '❌ Internal server error'
    }), 500


if __name__ == '__main__':
    logger.info('🚀 Starting Flask server...')
    logger.info(f'🌐 Server: http://{Config.HOST}:{Config.PORT}')
    logger.info(f'🐛 Debug mode: {Config.DEBUG}')
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
