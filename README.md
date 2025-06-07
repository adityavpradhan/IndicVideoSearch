# Environment Setup

 - Please install Python 3.12 if not available

 - Create a Virtual Env in the root directory
 ```bash
 python3.12 -m venv searchenv
 source searchenv/bin/activate
 ```

 - Install libraries
 ```bash
 pip3 install -r requirements.txt
 ```

 - Run the streamlit app
 ```bash
 streamlit run chat_app.py
 ```

 # Video RAG Pipeline
 The video RAG pipeline need not necessarily be part of the chat app. We can preprocess the data, store it in a vector DB and call the search functionality alone. This will keep the chat app separate from the video processing pipeline. RAG Pipeline code can be in separate python classes. We will just import search.

 # Video Summary Pipeline
 This will create a summary JSON file for the selected videos. Process the selected video (from the video folder) and Save the summary to the output folder. Each video is segmented into 10-second chunks, with The Video Summary Pipeline processes selected videos to create comprehensive JSON summary files stored in the output folder. Each video is automatically segmented into 10-second chunks, with both audio and visual content analyzed by Gemini AI. These summaries include precise timestamps, visual scene descriptions, spoken content transcriptions, and contextual information, all organized in  JSON format.

 Run Video Summary Pipline:
  ```bash
 python video_rag.py 
 ```     