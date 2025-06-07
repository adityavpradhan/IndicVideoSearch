import os
import streamlit as st
from typing import Dict, Any, Optional
from config import ChatAppConfig

class VideoUploadHandler:
    """Handles video upload functionality"""
    
    def __init__(self, config: ChatAppConfig):
        self.config = config
    
    def handle_upload(self) -> Optional[Dict[str, Any]]:
        """Handle video file upload"""
        st.sidebar.header("Video Upload")
        uploaded_video = st.sidebar.file_uploader(
            "Upload a video file", 
            type=self.config.supported_video_types
        )
        
        if uploaded_video:
            return self._save_uploaded_file(uploaded_video)
        return None
    
    def _save_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """Save uploaded file and return file info"""
        unique_filename = uploaded_file.name
        file_path = os.path.join(self.config.upload_folder, unique_filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.sidebar.success(f"Video '{uploaded_file.name}' uploaded successfully!")
        st.sidebar.info(f"Saved at: {file_path}")
        
        file_info = {
            "filename": uploaded_file.name,
            "path": file_path,
            "type": "video"
        }
        
        self._update_session_state(file_info)
        return file_info
    
    def _update_session_state(self, file_info: Dict[str, Any]):
        """Update session state with uploaded file info"""
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        st.session_state.uploaded_files.append(file_info)