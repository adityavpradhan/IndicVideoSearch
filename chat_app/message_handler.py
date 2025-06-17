import streamlit as st
from chat_app.query_transformation import QueryTransformer
from rag_pipeline.video_embedder import VideoEmbedder
from sarvamai.play import play, save
from llm_clients.sarvam_client import SarvamClient
# from llm_clients.gemini_client import GeminiClient
# from llm_clients.openai_client import OpenAIClient

class MessageHandler:
    """Handles chat message processing and display"""
    
    def __init__(self, llm, query_transformer: QueryTransformer, system_prompt: str = ""):
        self.llm = llm
        self.query_transformer = query_transformer
        self.system_prompt = system_prompt
        self.video_embedder = VideoEmbedder()  # Initialize video embedder for RAG
    
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
        st.write("Searching for relevant videos...")
        # I am calling RAG Pipeline here assuming that only one transformed query is generated. But 
        # Generate AI response with system prompt
        raw_results = self.video_embedder.search_videos(transformed_query)
        processed_results = self.process_search_results(raw_results)

        ai_response = self._generate_ai_response(transformed_query, processed_results, user_input)
        self.add_message("assistant", ai_response, "text")
    
    def process_audio_input(self, transcribed_text: str, audio_bytes: int):
        """Process audio input and generate AI response"""
        # Add audio info message
        self.add_message("user", f"🎤 Audio recorded ({audio_bytes} bytes)", "audio_info")
        
        if "Error" not in transcribed_text:
            st.success(f"Transcription (SarvamAI): {transcribed_text}")
            self.add_message("user", transcribed_text, "transcription")
            
            self.process_text_input(transcribed_text)
        else:
            st.error(f"Transcription failed: {transcribed_text}")
            self.add_message("assistant", f"Sorry, I couldn't understand the audio. {transcribed_text}", "error")
    
    def _generate_ai_response(self, transformed_query: str, processed_results, original_query: str) -> str:
        """Generate AI response using LLM with system prompt and append sources"""
        if self.llm:
            try:
                # Combine system prompt with user query
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": transformed_query + "Here's additional context for the user query. Answer his question based on this context:" + str(processed_results['context'])}
                ]
    
                sources = processed_results['sources']
                
                llm_response = self.llm.invoke(messages)
                
                # Extract the response content
                if isinstance(llm_response, str):
                    response_content = llm_response
                elif hasattr(llm_response, 'content'):
                    response_content = llm_response.content
                else:
                    response_content = f"Unexpected response format: {llm_response}"
                
                # Append sources to the response
                if sources:
                    response_content += "\n\n**Sources:**\n"
                    for i, source in enumerate(sources, 1):
                        response_content += f"{i}. {source}\n"
                
                return response_content
                
            except Exception as e:
                st.error(f"Error generating AI response: {e}")
                return f"Error generating response to: {original_query}"
        else:
            # For non-AI responses, still include sources if available
            base_response = f"NON AI Response to your query: {original_query}"
            return base_response
    
    def update_system_prompt(self, new_system_prompt: str):
        """Update the system prompt"""
        self.system_prompt = new_system_prompt
    
    def display_chat_history(self):
        """Display chat messages, and for assistant text add TTS controls."""
        st.markdown("---")
        st.subheader("Chat History")

        # instantiate TTS clients once
        sarvam = SarvamClient()
        # gemini = GeminiClient()
        # openai = OpenAIClient()

        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["type"] == "audio_info":
                    st.markdown(f"_{message['content']}_")
                elif message["type"] == "transcription":
                    st.markdown(f"🗣️ User (Transcribed): \"{message['content']}\"")
                elif message["type"] == "error":
                    st.error(message["content"])
                else:
                    # must be text
                    st.write(message["content"])

                    # only for assistant messages allow Text To Speech
                    if message["role"] == "assistant":
                        # create three columns for controls
                        c1, c2, c3 = st.columns([1,1,1], gap="small")
                        model = c1.selectbox(
                            "Model",
                            ["Sarvam"], # Commented out "OpenAI" and "Gemini" for now
                            key=f"tts_model_{idx}"
                        )
                        lang = c2.selectbox(
                            "Language",
                            ["en-IN", "hi-IN", "ta-IN", "kn-IN", "ml-IN"],
                            key=f"tts_lang_{idx}"
                        )
                        if c3.button("🔊 Play Audio", key=f"tts_play_{idx}"):
                            text = message["content"]
                            # call the right client
                            if model == "Sarvam":
                                audio_obj = sarvam.text_to_speech(
                                    text,
                                    voice="anushka",
                                    target_language_code=lang
                                )
                                # Sarvam returns a response object 
                                # with bytes under `.audio_content`
                                audio_bytes = audio_obj
                            elif model == "OpenAI":
                                buf = openai.text_to_speech(
                                    text,
                                    language_code=lang,
                                    voice="alloy"
                                )
                                audio_bytes = buf.getvalue()
                            else:  # Gemini
                                buf = gemini.text_to_speech(
                                    text,
                                    voice_name=f"{lang}-Standard-C"
                                )
                                audio_bytes = buf.getvalue()

                            # finally play
                            if model == "Sarvam":
                                play(audio_bytes)
                            else:
                                st.audio(audio_bytes, format="audio/wav")
    
    def clear_chat_history(self):
        """Clear chat history"""
        st.session_state.messages = []
    
    def format_time(self, seconds):
        """Convert seconds to MM:SS format"""
        try:
            seconds = float(seconds)
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        except:
            return "00:00"

    def process_search_results(self, results):
        """Process ChromaDB results into context and sources format"""
        if not results or not results.get('documents'):
            return {"context": [], "sources": []}
        
        context = []
        sources = []

        documents = results['documents'] # ChromaDB returns nested lists
        metadatas = results['metadatas']
        distances = results['distances']
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            # Add document content to context
            context.append(doc)
            
            # Format source information
            video_name = metadata.get('video_name', 'Unknown Video')
            start_time = metadata.get('start_time', '0')
            end_time = metadata.get('end_time', '0')
            
            # Format timestamps
            start_formatted = self.format_time(start_time)
            end_formatted = self.format_time(end_time)
            
            # Create source string
            source = f"{video_name} - [{start_formatted} - {end_formatted}]"
            sources.append(source)
        
        return {
            "context": context,
            "sources": sources
        }
