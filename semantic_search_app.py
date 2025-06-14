import streamlit as st
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# model options for the semantic search tool
NATIVE_MODELS = {
    "Muril (Google)": "google/muril-base-cased",
    "IndicSBERT-STS (l3cs)": "l3cube-pune/indic-sentence-similarity-sbert",
    "LaBSE": "sentence-transformers/LaBSE",
    "IndicBERT (ai4bharat)": "ai4bharat/IndicBERTv2-MLM-only",
    "Krutim Vyakyarth": "krutrim-ai-labs/Vyakyarth",
}
ENGLISH_MODELS = {
    "MiniLM (default)": "sentence-transformers/all-MiniLM-L6-v2",
    "BERT-base-nli": "sentence-transformers/bert-base-nli-mean-tokens",
    "DistilRoBERTa": "sentence-transformers/all-distilroberta-v1"
}

@st.cache_resource
def load_models(native_name, english_name):
    native_model = SentenceTransformer(NATIVE_MODELS[native_name])
    english_model = SentenceTransformer(ENGLISH_MODELS[english_name])
    translator = GoogleTranslator(source='auto', target='en') 
    return native_model, english_model, translator

def render_app():
    st.sidebar.header("Model Selection")
    selected_native_model = st.sidebar.selectbox("Native Language Model", list(NATIVE_MODELS.keys()))
    selected_english_model = st.sidebar.selectbox("English Embedding Model", list(ENGLISH_MODELS.keys()))

    native_model, english_model, translator = load_models(selected_native_model, selected_english_model)

    st.title("Semantic Search: Native Embedding vs. Translated Embedding")
    st.markdown("""
    Compare semantic similarity between a **query** and **corpus** entries using two methods:
    1.  **Native Embedding**: Directly embedding text in its original Indian language.
    2.  **Translated Embedding**: Translating text to English before embedding.
    """)

    query = st.text_input("Enter your query (in Hindi, Tamil, Malayalam, etc.): eg मैं सब्ज़ी लेने जा रहा हूँ")
    corpus_input = st.text_area(
        "Enter corpus (one sentence per line, in the same language):",
        "मैं बाज़ार जा रहा हूँ\nഞാൻ കടയിലേക്ക് പോകുന്നു\nநான் கடைக்கு போகிறேன்"
    )

    if st.button("Run Comparison"):
        if not query or not corpus_input:
            st.warning("Please enter both a query and a corpus.")
            return

        corpus = corpus_input.strip().split("\n")

        # --- Native Language Embedding ---
        st.markdown("### Native Language Embedding Result:")
        with st.spinner("Running native language embedding..."):
            chroma_native = chromadb.Client(Settings(anonymized_telemetry=False))
            embedding_fn_native = SentenceTransformerEmbeddingFunction(model_name=NATIVE_MODELS[selected_native_model])
            if "native" in [c.name for c in chroma_native.list_collections()]:
                chroma_native.delete_collection("native")
            collection_native = chroma_native.create_collection(name="native", embedding_function=embedding_fn_native)
            collection_native.add(documents=corpus, ids=[f"native_{i}" for i in range(len(corpus))])
            results_native = collection_native.query(query_texts=[query], n_results=len(corpus))
            for rank, (doc, dist) in enumerate(zip(results_native['documents'][0], results_native['distances'][0])):
                st.write(f"{rank+1}. {doc} (distance: {dist:.4f})")

        st.markdown("---")

        st.markdown("### Translated + English Embedding Result:")
        with st.spinner("Translating and running English embedding..."):
            try:
                def safe_translate(text):
                    result = translator.translate(text)
                    return result

                corpus_en = [safe_translate(text) for text in corpus]
                query_en = safe_translate(query)
            except Exception as e:
                st.error(f"Translation failed: {e}")
                corpus_en, query_en = [], ""

            if query_en:
                st.write(f"Query translated to: '{query_en}'")
                chroma_en = chromadb.Client(Settings(anonymized_telemetry=False))
                embedding_fn_en = SentenceTransformerEmbeddingFunction(model_name=ENGLISH_MODELS[selected_english_model])
                if "translated" in [c.name for c in chroma_en.list_collections()]:
                    chroma_en.delete_collection("translated")
                collection_en = chroma_en.create_collection(name="translated", embedding_function=embedding_fn_en)
                collection_en.add(documents=corpus_en, ids=[f"en_{i}" for i in range(len(corpus_en))])
                results_en = collection_en.query(query_texts=[query_en], n_results=len(corpus_en))
                for rank, (doc, dist) in enumerate(zip(results_en['documents'][0], results_en['distances'][0])):
                    original_doc = corpus[corpus_en.index(doc)]
                    st.write(f"{rank+1}. {original_doc} (translated: '{doc}') (distance: {dist:.4f})")

if __name__ == "__main__":
    render_app()