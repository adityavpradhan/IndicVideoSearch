import os

class ChatAppConfig:
    """Configuration class for the chat application"""
    
    def __init__(self):
        self.upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_videos")
        self.supported_video_types = ["mp4", "avi", "mov", "mkv"]
        self.llm_model = "gemini-2.0-flash-lite"
        self.llm_temperature = 0.7
        # Add default system prompt
        self.default_system_prompt = """You are a helpful AI assistant that specializes in video content analysis and search. 
You help users find information from their video collections. Be concise but informative in your responses.
When answering questions about videos, focus on the most relevant information. If the user input is not a question or unclear,
just say "I couldn't understand you. Could you please repeat your question?"."""
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.upload_folder, exist_ok=True)