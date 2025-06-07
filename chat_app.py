# chat_app.py
import streamlit as st
import os
import tempfile # For handling temporary audio files
from sarvam_client import SarvamClient # Import your SarvamClient
from audiorecorder import audiorecorder

# Import Langchain components
# from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(page_title="Chat Application", layout="wide")

# --- Authentication & Initialization ---
# Initialize SarvamClient
# This assumes SARVAMAI_API_KEY is set as an environment variable.
try:
    sarvam_client = SarvamClient()
    st.sidebar.success("SarvamAI Client Initialized.")
except ValueError as e:
    st.sidebar.error(f"Failed to initialize SarvamAI Client: {e}")
    st.stop() # Stop the app if client can't be initialized
except Exception as e:
    st.sidebar.error(f"An unexpected error occurred during SarvamAI Client initialization: {e}")
    st.stop()


# Initialize Langchain LLM for Query Transformation
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") # Set your Google API key
# llm = OpenAI(temperature=0.7)
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.7)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7)

st.sidebar.info("Langchain LLM would be initialized here.") # User feedback

# --- Langchain Query Transformation Setup ---
# This is where you would define your Langchain logic for query transformation.
# For Decomposition or HyDE, you might create specific prompts and chains.

# For Decomposition
decomposition_prompt_template = PromptTemplate(
    input_variables=["query"],
    template="Decompose the following user query into sub-queries: {query}"
)
decomposition_chain = LLMChain(llm=llm, prompt=decomposition_prompt_template, 
                               output_parser=StrOutputParser()) if llm else None

# for HyDE 
hyde_prompt_template = PromptTemplate(
    input_variables=["query"],
    template="Generate a hypothetical document that answers the query: {query}"
)
hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt_template, 
                      output_parser=StrOutputParser()) if llm else None

def transform_query_langchain(user_query, method="decomposition"):
    """
    Transforms the user query using Langchain (Decomposition or HyDE).
    This is a placeholder function. You need to implement the actual
    Langchain logic here.

    Args:
        user_query (str): The original query from the user.
        method (str): The transformation method ("decomposition" or "hyde").

    Returns:
        str: The transformed query. Returns original query if LLM not configured.
    """
    if not llm: # Check if LLM is initialized
        st.warning("Langchain LLM not configured. Skipping query transformation.")
        return user_query

    if method == "decomposition":
        if decomposition_chain:
            try:
                # transformed_query = decomposition_chain.invoke({"query": user_query})
                transformed_query = decomposition_chain.run(user_query)
                st.info(f"Decomposed Query: {transformed_query}")
                return transformed_query
            except Exception as e:
                st.error(f"Error during query decomposition: {e}")
                return user_query # Fallback to original query
        else:
            st.warning("Decomposition chain not initialized. Skipping transformation.")
            return user_query

    elif method == "hyde":
        if hyde_chain:
            try:
                # HyDE typically generates a document, you might use this document for retrieval
                # or extract keywords from it to form a new query.
                # hypothetical_document = hyde_chain.invoke({"query": user_query})
                hypothetical_document = hyde_chain.run(user_query)
                st.info(f"Hypothetical Document (HyDE): {hypothetical_document[:200]}...") # Show snippet
                # For this example, let's assume the document itself is the transformed query,
                # or you might have another step to process this document.
                return f"Query based on hypothetical document: {hypothetical_document}"
            except Exception as e:
                st.error(f"Error during HyDE generation: {e}")
                return user_query # Fallback
        else:
            st.warning("HyDE chain not initialized. Skipping transformation.")
            return user_query
    else:
        # Default to original query if method is not recognized or LLM/chain not set up
        return user_query

# --- Session State Management ---
# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with video upload
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_videos")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.sidebar.title("Options")
st.sidebar.header("Video Upload")
uploaded_video = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"]) # We can restrict ro only mp4 to start with

if uploaded_video:
    # st.sidebar.video(uploaded_video)
    unique_filename = uploaded_video.name
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # Save the file to the designated folder
    with open(file_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    
    # Show success message with file path
    st.sidebar.success(f"Video '{uploaded_video.name}' uploaded successfully!")
    st.sidebar.info(f"Saved at: {file_path}")
    
    # Store the file path in session state for later access by LLMs
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    st.session_state.uploaded_files.append({
        "filename": uploaded_video.name,
        "path": file_path,
        "type": "video"
    })
    st.sidebar.success(f"Video '{uploaded_video.name}' uploaded successfully!")


# --- Main Chat Panel ---
st.title("🗣️🎙️ Chat with AI ")
st.markdown("---")

# --- UI for Inputs (Text and Audio) ---
# Use columns to place text input and audio recorder side-by-side (or as close as possible)
col1, col2 = st.columns([4, 1]) # Adjust column widths as needed

with col1:
    user_text_input = st.chat_input("Type a message or record audio...")

with col2:
    # NEEDED TO INSTALL ffmpeg
    # brew install ffmpeg for MACOS
    # apt-get install ffmpeg for Ubuntu

    recorded_audio = audiorecorder(start_prompt="start recording",
                                   stop_prompt="stop recording",
                                   pause_prompt="pause",
                                   show_visualizer=True,
                                   start_style={'color': 'green', 'font-weight': 'bold'},
                                   stop_style={'color': 'red', 'font-weight': 'bold'},
                                   pause_style={'color': 'blue', 'font-weight': 'bold'})

    if len(recorded_audio) > 0:
        # To play audio in frontend:
        # st.audio(recorded_audio.export().read())  

        # To save audio to a file, use pydub export method:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            recorded_audio.export(tmp_audio_file.name, format="wav")
            st.success(f"Audio recorded ({len(recorded_audio)} bytes)")
            tmp_audio_file_path = tmp_audio_file.name
            st.audio(tmp_audio_file.name, format="audio/wav")

        # To get audio properties, use pydub AudioSegment properties:
        st.write(f"Frame rate: {recorded_audio.frame_rate}, "
                 f"Frame width: {recorded_audio.frame_width}, "
                 f"Duration: {recorded_audio.duration_seconds} seconds")


# --- Process Inputs and Generate Responses ---

# 1. Handle Text Input
if user_text_input:
    st.session_state.messages.append({"role": "user", "content": user_text_input, "type": "text"})

    # Query Transformation for text input

    transformed_text_query = transform_query_langchain(user_text_input, method="decomposition") # or "hyde"
    # st.write(f"Transformed Text Query (Langchain): {transformed_text_query["query"]}")
    st.write(f"Transformed Text Query (Langchain): {transformed_text_query}")
    # For now, we use the original query for the AI response placeholder.

    # Use the transformed query for the AI response
    # ai_response_text = llm.invoke(transformed_text_query["query"]) if llm else f"NON AI Response to your query: {user_text_input}"
    ai_response_text = llm.invoke(transformed_text_query) if llm else f"NON AI Response to your query: {user_text_input}"
    st.session_state.messages.append({"role": "assistant", "content": ai_response_text, "type": "text"})
    st.rerun() # Rerun to update the chat display immediately

# 2. Handle Audio Input
if recorded_audio:
    st.info("Audio recorded! Processing...")
    print("Got recorded audio")

    # Add a message to the chat indicating audio was received
    st.session_state.messages.append({
        "role": "user",
        "content": f"🎤 Audio recorded ({len(recorded_audio)} bytes)",
        "type": "audio_info" # Custom type for displaying audio info
    })

    # Transcribe the audio using SarvamClient
    transcribed_text = sarvam_client.speech_to_text(tmp_audio_file_path)
    print("Got transcribed text")

    # Clean up the temporary file
    try:
        os.remove(tmp_audio_file_path)
        print("Temporary audio file deleted")
    except Exception as e:
        st.warning(f"Could not delete temporary audio file: {e}")

    if "Error" not in transcribed_text:
        st.success(f"Transcription (SarvamAI): {transcribed_text}")
        # Add transcribed text to chat as a user message
        st.session_state.messages.append({"role": "user", "content": transcribed_text, "type": "transcription"})

        # Now, use Langchain to transform the transcribed query
        # You can choose the method, e.g., "decomposition" or "hyde"
        transformed_query = transform_query_langchain(transcribed_text, method="decomposition")
        st.info(f"Transformed Query (Langchain): {transformed_query}")

        # For now, we'll use the transcribed text directly for the AI response
        # In a full application, you'd use the `transformed_query`
        print("Calling LLM with transformed query")
        ai_response_audio = llm.invoke(transformed_query) if llm else f"NON AI Response to your query: {transcribed_text}"
        print("Got AI response", ai_response_audio)
        # ai_response_audio = llm.invoke(transformed_query["query"]) if llm else f"NON AI Response to your query: {transcribed_text}"
        st.session_state.messages.append({"role": "assistant", "content": ai_response_audio, "type": "text"})

    else:
        # Handle transcription error
        st.error(f"Transcription failed: {transcribed_text}")
        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I couldn't understand the audio. {transcribed_text}", "type": "error"})

    st.rerun()


# --- Display Chat Messages ---
st.markdown("---")
st.subheader("Chat History")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "audio_info":
            st.markdown(f"_{message['content']}_") # Italicize audio info
        elif message["type"] == "transcription":
            st.markdown(f"🗣️ User (Transcribed): \"{message['content']}\"")
        elif message["type"] == "error":
            st.error(message["content"])
        else: # Default to text content
            st.write(message["content"])

# Make sure that the chat autoscrolls to the bottom
js_command = "window.scrollTo(0, document.body.scrollHeight);"
st.markdown(f"<script>{js_command}</script>", unsafe_allow_html=True)

# --- Sidebar Information ---
st.sidebar.header("Controls & Info")
st.sidebar.markdown("""
This chat application allows you to interact via text or voice to search your videos.
- Type a message in the input box.
- Click the microphone icon to record audio.
- Audio is transcribed using **SarvamAI**.
- Use responsibly and have fun!
""")

# Clear chat history if "Clear Chat" button is clicked
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
