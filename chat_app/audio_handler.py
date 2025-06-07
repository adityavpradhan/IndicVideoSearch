import os
import tempfile
import streamlit as st
from audiorecorder import audiorecorder

class AudioHandler:
    """Handles audio recording and processing"""
    
    def __init__(self, sarvam_client):
        self.sarvam_client = sarvam_client
    
    def record_audio(self):
        """Handle audio recording UI"""
        recorded_audio = audiorecorder(
            start_prompt="start recording",
            stop_prompt="stop recording",
            pause_prompt="pause",
            show_visualizer=True,
            start_style={'color': 'green', 'font-weight': 'bold'},
            stop_style={'color': 'red', 'font-weight': 'bold'},
            pause_style={'color': 'blue', 'font-weight': 'bold'}
        )
        
        if len(recorded_audio) > 0:
            # Check if this is a new recording by comparing with previous state
            if self._is_new_recording(recorded_audio):
                return self._process_recorded_audio(recorded_audio)
        return None, None
    
    def _is_new_recording(self, recorded_audio):
        """Check if this is a new recording to avoid reprocessing"""
        current_audio_hash = hash(str(len(recorded_audio)) + str(recorded_audio.duration_seconds))
        
        if "last_audio_hash" not in st.session_state:
            st.session_state.last_audio_hash = None
        
        if st.session_state.last_audio_hash != current_audio_hash:
            st.session_state.last_audio_hash = current_audio_hash
            return True
        return False
    
    def _process_recorded_audio(self, recorded_audio):
        """Process recorded audio and return file path and info"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            recorded_audio.export(tmp_audio_file.name, format="wav")
            st.success(f"Audio recorded ({len(recorded_audio)} bytes)")
            st.audio(tmp_audio_file.name, format="audio/wav")
            
            st.write(f"Frame rate: {recorded_audio.frame_rate}, "
                    f"Frame width: {recorded_audio.frame_width}, "
                    f"Duration: {recorded_audio.duration_seconds} seconds")
            
            return tmp_audio_file.name, len(recorded_audio)
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio using SarvamAI"""
        if not self.sarvam_client:
            return "Error: SarvamAI client not initialized"
        
        try:
            transcribed_text = self.sarvam_client.speech_to_text(audio_file_path)
            return transcribed_text
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            # Clean up temporary file
            try:
                os.remove(audio_file_path)
            except Exception as e:
                st.warning(f"Could not delete temporary audio file: {e}")