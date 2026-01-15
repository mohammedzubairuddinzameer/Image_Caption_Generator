import requests
import base64
from io import BytesIO
from PIL import Image
import logging
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageCaptioner:
    """
    Main image captioning engine using BLIP model
    Simple: Ye class images ko captions dene ka main kaam karti hai
    Hyderabadi: Ye class images pe caption lagane ka main kaam karte hai
    """
    
    def __init__(self, token=None):
        """
        Initialize the captioner with optional HF token
        
        Args:
            token (str): Hugging Face API token (optional)
        
        Simple: Captioner ko start karte hain with API token
        Hyderabadi: Captioner ko shuru karte hain API token ke saath
        """
        self.model_name = Config.HF_MODEL_NAME
        self.api_url = Config.HF_API_URL
        self.headers = {}
        
        # Set authorization header if token provided
        if token or Config.HF_TOKEN:
            self.headers['Authorization'] = f'Bearer {token or Config.HF_TOKEN}'
            logger.info('✅ ImageCaptioner initialized with authentication')
        else:
            logger.warning('⚠️ No HF_TOKEN found - API calls may be rate limited')
        
        logger.info(f'✅ Model: {self.model_name}')
        logger.info(f'✅ API URL: {self.api_url}')
    
    def caption_from_url(self, image_url):
        """
        Generate caption from image URL
        
        Args:
            image_url (str): URL of the image
        
        Returns:
            dict: {'status': str, 'caption': str, 'message': str}
        
        Example:
            result = captioner.caption_from_url('https://example.com/image.jpg')
        """
        try:
            logger.info(f'📥 Downloading image from URL: {image_url}')
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            image_bytes = response.content
            return self._caption_from_bytes(image_bytes)
            
        except requests.RequestException as e:
            logger.error(f'❌ URL download failed: {str(e)}')
            return {
                'status': 'error',
                'caption': None,
                'message': f'❌ Failed to download image: {str(e)}'
            }
    
    def caption_from_file(self, image_path):
        """
        Generate caption from local file
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            dict: {'status': str, 'caption': str, 'message': str}
        """
        try:
            logger.info(f'📂 Reading image from file: {image_path}')
            
            # Validate and open image
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)  # Reopen after verify
            
            # Convert to bytes
            buffer = BytesIO()
            img.save(buffer, format=img.format or 'PNG')
            image_bytes = buffer.getvalue()
            
            return self._caption_from_bytes(image_bytes)
            
        except FileNotFoundError:
            logger.error(f'❌ File not found: {image_path}')
            return {
                'status': 'error',
                'caption': None,
                'message': f'❌ File not found: {image_path}'
            }
        except Exception as e:
            logger.error(f'❌ File processing failed: {str(e)}')
            return {
                'status': 'error',
                'caption': None,
                'message': f'❌ Invalid image file: {str(e)}'
            }
    
    def caption_from_pil(self, pil_image):
        """
        Generate caption from PIL Image object
        
        Args:
            pil_image (PIL.Image): PIL Image object
        
        Returns:
            dict: {'status': str, 'caption': str, 'message': str}
        """
        try:
            logger.info('🖼️ Processing PIL Image')
            
            if not isinstance(pil_image, Image.Image):
                raise ValueError('Input is not a valid PIL Image')
            
            # Convert to bytes
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            
            return self._caption_from_bytes(image_bytes)
            
        except Exception as e:
            logger.error(f'❌ PIL image processing failed: {str(e)}')
            return {
                'status': 'error',
                'caption': None,
                'message': f'❌ Image processing failed: {str(e)}'
            }
    
    def _caption_from_bytes(self, image_bytes):
        """
        Internal method to call Hugging Face API
        
        Args:
            image_bytes (bytes): Image data in bytes
        
        Returns:
            dict: {'status': str, 'caption': str, 'message': str}
        """
        try:
            logger.info('🚀 Sending request to Hugging Face API...')
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=image_bytes,
                timeout=30
            )
            
            # Handle different response codes
            if response.status_code == 200:
                result = response.json()
                
                # Extract caption from response
                if isinstance(result, list) and len(result) > 0:
                    caption = result[0].get('generated_text', '')
                elif isinstance(result, dict):
                    caption = result.get('generated_text', '')
                else:
                    caption = str(result)
                
                logger.info(f'✅ Caption generated: "{caption}"')
                return {
                    'status': 'success',
                    'caption': caption,
                    'message': f'✅ Caption generated successfully!\n\n💬 "{caption}"'
                }
            
            elif response.status_code == 503:
                logger.warning('⚠️ Model is loading, please retry')
                return {
                    'status': 'warning',
                    'caption': None,
                    'message': '⚠️ Model is loading, please wait and try again'
                }
            
            else:
                error_msg = response.text[:200]
                logger.error(f'❌ API error: {response.status_code}')
                return {
                    'status': 'error',
                    'caption': None,
                    'message': f'❌ API Error {response.status_code}: {error_msg}'
                }
        
        except requests.Timeout:
            logger.error('❌ Request timeout')
            return {
                'status': 'error',
                'caption': None,
                'message': '❌ Request timed out. Please try again.'
            }
        
        except Exception as e:
            logger.error(f'❌ Unexpected error: {str(e)}')
            return {
                'status': 'error',
                'caption': None,
                'message': f'❌ Unexpected error: {str(e)}'
            }


# Create global instance (optional)
def create_captioner(token=None):
    """Factory function to create ImageCaptioner instance"""
    return ImageCaptioner(token=token)
