# Indic Video Search App and Comparative Analysis of Language Models for Key components of the RAG pipeline

## Environment Setup

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
```
 - Install libraries
 ```bash
 sudo apt install ffmpeg
 pip3 install -r requirements.txt
 ```

- Process Some videos first. You will need the GEMINI_API_KEY to do this step. Otherwise you can use the already available chroma db in the chroma_db folder.
```bash
# There is a sample video in the videos folder. You can add more videos here. THen run
rm -rf chroma_db/chroma.sqlite3 #Optional step. There is a preprocessed db here. You can start fresh if you delete this d

python3 process_videos.py #Follow the interactive mode
```

 - Run the streamlit app
 ```bash
 streamlit run main_app.py
 (or)
 streamlit run main_app.py --server.fileWatcherType none #If you see some torch related errors in the terminal. You will have to manually restart the streamlit app from terminal after every file change in this case as the run & rerun commands won't be available
 ```

## Documentation

For a detailed analysis of our search approaches and performance metrics, please refer to our [Comparative Analysis Paper](docs/Comparative_Analysis_Paper.pdf).

## App Screenshots:

Our Chat App supports both Text & Voice inputs.

### Text input
![text Input](docs/Chat_app_with_text_input.png)

You can also ask questions in code-mixed language. There are a few sample questions given in the main app page for you to try out:

![code-mixed Input](docs/multilingual_input.png)
### Voice input & Voice Output
![Voice Input](docs/Chat_app_with_voice_interaction.png)

### Embedding Model Comparison Companion App
The  Embedding Model Comparison Companion app can be accessed from the dropdown menu in the top left. It allows you to feed in a multilingual data corpus and a query. A semantic similarity comparison between a query and corpus entries is done and displayed using two methods:
  - Native Embedding: Directly embedding text in its original Indian language.
  - Translated Embedding: Translating text to English before embedding.
More details about this companion app and its use can be found in the above paper
![Semantic Comparison App](docs/Semantic_search_app.png)
