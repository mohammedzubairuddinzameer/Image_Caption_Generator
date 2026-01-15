import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class for the Image Captioning Chatbot
    Simple English: Ye class saari settings ko ek jagah rakhti hai
    Roman Hyderabadi: Ye class mein saari settings ek saath rakhte hain
    """
    
    # Hugging Face API Settings
    HF_TOKEN = os.getenv('HF_TOKEN', '')
    HF_MODEL_NAME = "Salesforce/blip-image-captioning-base"
    HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}"
    
    # Flask Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    
    # Server Settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    @staticmethod
    def allowed_file(filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
