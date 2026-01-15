import logging
from datetime import datetime
from caption_engine import ImageCaptioner

logger = logging.getLogger(__name__)


class ChatbotInterface:
    """
    Interactive chatbot interface for image captioning
    
    Simple: Ye chatbot users se interact karne aur images ko caption dene ke liye hai
    Hyderabadi: Ye chatbot users se baat karne aur images pe caption lagane ke liye hai
    
    Features:
    - Upload images (file, URL, PIL)
    - Generate captions
    - Track conversation history
    - Error handling with user-friendly messages
    """
    
    def __init__(self, captioner=None):
        """
        Initialize chatbot with captioner instance
        
        Args:
            captioner (ImageCaptioner): Optional captioner instance
        """
        self.captioner = captioner or ImageCaptioner()
        self.conversation_history = []
        logger.info('🤖 Image Captioning Chatbot initialized!')
    
    def process_image_upload(self, image_input, input_type='file', user_id=None):
        """
        Process uploaded image and generate caption
        
        Args:
            image_input: Image file path, URL, or PIL Image
            input_type (str): Type of input - 'file', 'url', or 'pil'
            user_id (str): Optional user identifier for tracking
        
        Returns:
            dict: {
                'status': 'success'|'warning'|'error',
                'caption': str or None,
                'message': str,
                'timestamp': str,
                'input_type': str
            }
        
        Example (Simple): 
            result = chatbot.process_image_upload('image.jpg', 'file')
        
        Example (Hyderabadi):
            result = chatbot.process_image_upload('image.jpg', 'file')
            print(result['message'])  # User-friendly message
        """
        timestamp = datetime.now().isoformat()
        
        try:
            logger.info(f'📥 Processing {input_type} upload...')
            
            # Route to appropriate captioning method
            if input_type == 'file':
                result = self.captioner.caption_from_file(image_input)
            elif input_type == 'url':
                result = self.captioner.caption_from_url(image_input)
            elif input_type == 'pil':
                result = self.captioner.caption_from_pil(image_input)
            else:
                return {
                    'status': 'error',
                    'caption': None,
                    'message': f'❌ Unsupported input type: {input_type}',
                    'timestamp': timestamp,
                    'input_type': input_type
                }
            
            # Add metadata to result
            result['timestamp'] = timestamp
            result['input_type'] = input_type
            if user_id:
                result['user_id'] = user_id
            
            # Save to history if successful
            if result['status'] == 'success':
                self.conversation_history.append(result)
                logger.info(f'✅ Caption saved to history (Total: {len(self.conversation_history)})')
            
            return result
        
        except Exception as e:
            logger.error(f'❌ Processing failed: {str(e)}')
            return {
                'status': 'error',
                'caption': None,
                'message': f'❌ Processing failed: {str(e)}',
                'timestamp': timestamp,
                'input_type': input_type
            }
    
    def get_conversation_history(self, user_id=None, limit=None):
        """
        Get conversation history
        
        Args:
            user_id (str): Optional filter by user ID
            limit (int): Maximum number of entries to return
        
        Returns:
            list: List of conversation entries
        
        Example:
            history = chatbot.get_conversation_history(limit=10)
        """
        history = self.conversation_history
        
        # Filter by user_id if provided
        if user_id:
            history = [entry for entry in history if entry.get('user_id') == user_id]
        
        # Apply limit if provided
        if limit:
            history = history[-limit:]
        
        return history
    
    def clear_history(self, user_id=None):
        """
        Clear conversation history
        
        Args:
            user_id (str): If provided, only clear history for this user
        
        Returns:
            dict: Status message
        """
        if user_id:
            initial_count = len(self.conversation_history)
            self.conversation_history = [
                entry for entry in self.conversation_history 
                if entry.get('user_id') != user_id
            ]
            cleared_count = initial_count - len(self.conversation_history)
            message = f'🗑️ Cleared {cleared_count} entries for user {user_id}'
        else:
            cleared_count = len(self.conversation_history)
            self.conversation_history = []
            message = f'🗑️ Cleared all {cleared_count} conversation entries'
        
        logger.info(message)
        return {'status': 'success', 'message': message}
    
    def get_stats(self):
        """
        Get chatbot statistics
        
        Returns:
            dict: Statistics about usage
        """
        total = len(self.conversation_history)
        successful = sum(1 for entry in self.conversation_history if entry['status'] == 'success')
        
        return {
            'total_requests': total,
            'successful_captions': successful,
            'success_rate': f'{(successful/total*100):.1f}%' if total > 0 else 'N/A'
        }


# Factory function
def create_chatbot(token=None):
    """Create ChatbotInterface instance with custom token"""
    captioner = ImageCaptioner(token=token)
    return ChatbotInterface(captioner=captioner)
