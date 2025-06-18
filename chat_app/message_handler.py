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
        """Process text input and generate AI response using RAG-fusion"""
        try:
            # Add user message
            self.add_message("user", user_input, "text")
            
            # Generate diverse queries using RAG-fusion
            transformed_queries = self.query_transformer.transform_query(user_input, "rag_fusion")
            if not transformed_queries:
                transformed_queries = [user_input]
            
            # Initialize combined results
            combined_results = {
                'documents': [],
                'metadatas': [],
                'distances': []
            }
            
            # Search with each query variation
            st.write("Performing RAG-fusion search...")
            success_count = 0
            
            with st.spinner("Searching across multiple perspectives..."):
                for i, query in enumerate(transformed_queries, 1):
                    try:
                        results = self.video_embedder.search_videos(query, n_results=3)
                        
                        if results and isinstance(results, dict) and 'documents' in results:
                            combined_results['documents'].extend(results['documents'])
                            combined_results['metadatas'].extend(results['metadatas'])
                            combined_results['distances'].extend(results['distances'])
                            success_count += 1
                    except Exception as e:
                        st.warning(f"Search failed for query variation {i}: {str(e)}")
                        continue
            
            # Handle no results case
            if success_count == 0:
                st.warning("No relevant video content found.")
                self.add_message("assistant", 
                               "I couldn't find any relevant video content. Please try rephrasing your question.", 
                               "text")
                return
                
            # Process and generate response
            processed_results = self.process_search_results(combined_results)
            
            # Generate comprehensive response
            st.write("Generating response...")
            ai_response = self._generate_ai_response(transformed_queries, processed_results, user_input)
            self.add_message("assistant", ai_response, "text")
            
        except Exception as e:
            error_msg = f"Error processing your request: {str(e)}"
            st.error(error_msg)
            self.add_message("assistant", error_msg, "error")
    
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
    
    def _generate_ai_response(self, transformed_queries: list, processed_results: dict, original_query: str) -> str:
        if not self.llm:
            return f"NON AI Response to your query: {original_query}"

        try:
            history = []
            messages = st.session_state.get("messages", [])
            pairs = []
            i = 0
            while i < len(messages) - 1:
                if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                    pairs.append((messages[i]["content"], messages[i + 1]["content"]))
                i += 1
            last_10_pairs = pairs[-10:]

            for user_msg, assistant_msg in last_10_pairs:
                history.append({"role": "user", "content": user_msg})
                history.append({"role": "assistant", "content": assistant_msg})

            query_context = original_query
            if transformed_queries and len(transformed_queries) > 1:
                query_context += "\n\nI explored this question from multiple perspectives:\n"
                for i, q in enumerate(transformed_queries, 1):
                    query_context += f"{i}. {q}\n"

            context_blob = "\n".join(processed_results['context'])
            prompt = query_context + "\n\nHere's relevant context from the videos:\n" + context_blob

            full_messages = [{"role": "system", "content": self.system_prompt}] + history + [{"role": "user", "content": prompt}]
            st.write("📨 Prompt to LLM:", prompt[:500] + "..." if len(prompt) > 500 else prompt)

            llm_response = self.llm.invoke(full_messages)

            if isinstance(llm_response, str):
                response_content = llm_response
            elif hasattr(llm_response, 'content'):
                response_content = llm_response.content
            else:
                response_content = f"Unexpected response format: {llm_response}"

            sources = processed_results.get('sources', [])
            if sources:
                response_content += "\n\n**Sources:**\n"
                for i, source in enumerate(sources, 1):
                    response_content += f"{i}. {source}\n"

            return response_content

        except Exception as e:
            error_msg = f"Error generating AI response: {e}"
            st.error(error_msg)
            print(error_msg)
            return f"Error generating response to: {original_query}"
    
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
        """Process ChromaDB results into context and sources format with deduplication, limiting to top 5 results"""
        if not results or not isinstance(results, dict) or 'documents' not in results:
            return {"context": [], "sources": []}
        
        # Create a list to store all results with their distances for sorting
        all_results = []
        seen_content = set()  # Track unique content
        
        try:
            # Prepare data
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])
            distances = results.get('distances', [])
            
            # Ensure all lists have the same length
            min_length = min(len(documents), len(metadatas), len(distances))
            
            # Combine all results with their distances
            for i in range(min_length):
                doc = documents[i]
                metadata = metadatas[i]
                distance = distances[i]
                
                # Skip empty or duplicate content
                if not doc or doc in seen_content:
                    continue
                seen_content.add(doc)
                
                video_name = metadata.get('video_name', 'Unknown Video')
                start_time = metadata.get('start_time', '0')
                end_time = metadata.get('end_time', '0')
                
                try:
                    start_formatted = self.format_time(start_time)
                    end_formatted = self.format_time(end_time)
                    source = f"{video_name} [{start_formatted} - {end_formatted}]"
                except Exception as e:
                    print(f"Error formatting source: {e}")
                    source = f"{video_name} [timestamp error]"
                
                all_results.append({
                    'content': doc.strip(),
                    'source': source,
                    'distance': distance
                })
            
            # Sort by distance (lower is better) and take top 5
            all_results.sort(key=lambda x: x['distance'])
            top_results = all_results[:5]
            
            # Separate into context and sources
            context = [r['content'] for r in top_results]
            sources = [r['source'] for r in top_results]
            
            return {
                "context": context,
                "sources": sources
            }
            
        except Exception as e:
            print(f"Error processing search results: {e}")
            return {"context": [], "sources": []}
