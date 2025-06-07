# Environment Setup

 - Please install Python 3.12 if not available

 - Create a Virtual Env in the root directory
 ```bash
 python3.12 -m venv searchenv
 source searchenv/bin/activate
 ```

- Add all relevant API keys
```bash
export SARVAMAI_API_KEY="<YOUR-KEY>"
export GEMINI_API_KEY="<YOUR-KEY>"
export GOOGLE_API_KEY="<SAME KEY>" #Code is referencing it in 2 different ways. Will fix
```
 - Install libraries
 ```bash
 sudo apt install ffmpeg
 pip3 install -r requirements.txt
 ```

 - Run the streamlit app
 ```bash
 streamlit run chat_app.py
 ```

 # Video RAG Pipeline
 The video RAG pipeline need not necessarily be part of the chat app. We can preprocess the data, store it in a vector DB and call the search functionality alone. This will keep the chat app separate from the video processing pipeline. RAG Pipeline code can be in separate python classes. We will just import search.
