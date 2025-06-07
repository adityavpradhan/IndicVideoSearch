import streamlit as st
from chat_app.query_transformation import QueryTransformer

class MessageHandler:
    """Handles chat message processing and display"""
    
    def __init__(self, llm, query_transformer: QueryTransformer, system_prompt: str = ""):
        self.llm = llm
        self.query_transformer = query_transformer
        self.system_prompt = system_prompt
    
    def add_message(self, role: str, content: str, msg_type: str = "text"):
        """Add message to session state"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        st.session_state.messages.append({
            "role": role,
            "content": content,
            "type": msg_type
        })
    
    def process_text_input(self, user_input: str):
        """Process text input and generate AI response"""
        self.add_message("user", user_input, "text")
        
        # Transform query
        transformed_query = self.query_transformer.transform_query(user_input, "None")
        st.write(f"Transformed Text Query (Langchain): {transformed_query}")
        
        # Generate AI response with system prompt
        ai_response = self._generate_ai_response(transformed_query, user_input)
        self.add_message("assistant", ai_response, "text")
    
    def process_audio_input(self, transcribed_text: str, audio_bytes: int):
        """Process audio input and generate AI response"""
        # Add audio info message
        self.add_message("user", f"🎤 Audio recorded ({audio_bytes} bytes)", "audio_info")
        
        if "Error" not in transcribed_text:
            st.success(f"Transcription (SarvamAI): {transcribed_text}")
            self.add_message("user", transcribed_text, "transcription")
            
            # Transform query
            transformed_query = self.query_transformer.transform_query(transcribed_text, "None")
            st.info(f"Transformed Query (Langchain): {transformed_query}")
            
            # Generate AI response with system prompt
            ai_response = self._generate_ai_response(transformed_query, transcribed_text)
            self.add_message("assistant", ai_response, "text")
        else:
            st.error(f"Transcription failed: {transcribed_text}")
            self.add_message("assistant", f"Sorry, I couldn't understand the audio. {transcribed_text}", "error")
    
    def _generate_ai_response(self, transformed_query: str, original_query: str) -> str:
        """Generate AI response using LLM with system prompt"""
        if self.llm:
            try:
                # Combine system prompt with user query
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": transformed_query}
                ]
                
                llm_response = self.llm.invoke(messages)
                if isinstance(llm_response, str):
                    return llm_response
                elif hasattr(llm_response, 'content'):
                    return llm_response.content
                else:
                    return f"Unexpected response format: {llm_response}"
            except Exception as e:
                st.error(f"Error generating AI response: {e}")
                return f"Error generating response to: {original_query}"
        else:
            return f"NON AI Response to your query: {original_query}"
    
    def update_system_prompt(self, new_system_prompt: str):
        """Update the system prompt"""
        self.system_prompt = new_system_prompt
    
    def display_chat_history(self):
        """Display chat messages"""
        st.markdown("---")
        st.subheader("Chat History")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["type"] == "audio_info":
                    st.markdown(f"_{message['content']}_")
                elif message["type"] == "transcription":
                    st.markdown(f"🗣️ User (Transcribed): \"{message['content']}\"")
                elif message["type"] == "error":
                    st.error(message["content"])
                else:
                    st.write(message["content"])
    
    def clear_chat_history(self):
        """Clear chat history"""
        st.session_state.messages = []