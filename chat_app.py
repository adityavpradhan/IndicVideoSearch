import streamlit as st
import os
import sarvamai
# Set page config
st.set_page_config(page_title="Chat Application", layout="wide")

# Initialize session state for chat history
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

# Main chat panel
st.title("Chat Panel")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Text input through chat input
user_input = st.chat_input("Type a message...")

# We can similarly get audio input from the user and then call the SarvamAI API
# to convert speech from any language (they have automatic language identification) to text in English.

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Here you would typically call our LLM, SarvamAI APIs, RAG modules to get a response
    st.rerun()