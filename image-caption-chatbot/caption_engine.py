import requests
from io import BytesIO
from PIL import Image
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageCaptioner:
    def __init__(self, token=None):
        self.model_name = Config.HF_MODEL_NAME
        self.api_url = Config.HF_API_URL
        self.headers = {}
        
        if token or Config.HF_TOKEN:
            self.headers['Authorization'] = f'Bearer {token or Config.HF_TOKEN}'
            logger.info('✅ ImageCaptioner initialized with auth')
        else:
            logger.warning('⚠️ No HF_TOKEN found')
        
    def caption_from_file(self, image_path):
        try:
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)
            buffer = BytesIO()
            img.save(buffer, format=img.format or 'PNG')
            return self._caption_from_bytes(buffer.getvalue())
        except Exception as e:
            return {'status': 'error', 'caption': None, 'message': f'❌ {str(e)}'}
    
    def caption_from_pil(self, pil_image):
        try:
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            return self._caption_from_bytes(buffer.getvalue())
        except Exception as e:
            return {'status': 'error', 'caption': None, 'message': f'❌ {str(e)}'}
    
    def _caption_from_bytes(self, image_bytes):
        try:
            response = requests.post(self.api_url, headers=self.headers, data=image_bytes, timeout=30)
            if response.status_code == 200:
                result = response.json()
                caption = result[0].get('generated_text', '') if isinstance(result, list) else str(result)
                return {'status': 'success', 'caption': caption, 'message': f'✅ "{caption}"'}
            return {'status': 'error', 'caption': None, 'message': f'❌ API Error {response.status_code}'}
        except Exception as e:
            return {'status': 'error', 'caption': None, 'message': f'❌ {str(e)}'}
