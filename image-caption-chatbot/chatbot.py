import logging
from datetime import datetime
from caption_engine import ImageCaptioner

logger = logging.getLogger(__name__)

class ChatbotInterface:
    def __init__(self, captioner=None):
        self.captioner = captioner or ImageCaptioner()
        self.conversation_history = []
        logger.info('🤖 Chatbot initialized')
    
    def process_image_upload(self, image_input, input_type='file'):
        timestamp = datetime.now().isoformat()
        try:
            if input_type == 'file':
                result = self.captioner.caption_from_file(image_input)
            elif input_type == 'pil':
                result = self.captioner.caption_from_pil(image_input)
            else:
                return {'status': 'error', 'caption': None, 'message': f'❌ Invalid type', 'timestamp': timestamp}
            
            result['timestamp'] = timestamp
            result['input_type'] = input_type
            
            if result['status'] == 'success':
                self.conversation_history.append(result)
            
            return result
        except Exception as e:
            return {'status': 'error', 'caption': None, 'message': f'❌ {str(e)}', 'timestamp': timestamp}
    
    def get_stats(self):
        total = len(self.conversation_history)
        successful = sum(1 for e in self.conversation_history if e['status'] == 'success')
        return {
            'total_requests': total,
            'successful_captions': successful,
            'success_rate': f'{(successful/total*100):.1f}%' if total > 0 else 'N/A'
        }
