from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import json

def test_persistence():
    print("\nTesting ChromaDB Persistence")
    print("-" * 25)
    
    # Initialize the embedding function - must match what was used to create the store
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the existing Chroma store
    print("\nLoading vectorstore from chroma_db...")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_function,
        collection_name="video_summaries"
    )
    
    # Get collection info
    try:
        collection = vectorstore._collection
        print(f"\nCollection Stats:")
        print(f"Name: {collection.name}")
        print(f"Count: {collection.count()}")
    except Exception as e:
        print(f"Error getting collection info: {e}")
    
    # Test specific queries
    queries = [
        "How does backpropagation use the chain rule?",
        "How are weights updated in a neural network?",
        "What is the formula for the loss function?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * (len(query) + 7))
        try:
            results = vectorstore.similarity_search(query, k=2)
            
            for i, doc in enumerate(results):
                print(f"\nMatch {i+1}:")
                print(f"Start time: {doc.metadata.get('start_time', 'N/A')} seconds")
                print(f"End time: {doc.metadata.get('end_time', 'N/A')} seconds")
                print(f"Content: {doc.page_content[:300]}...")
        except Exception as e:
            print(f"Error during search: {e}")

if __name__ == "__main__":
    test_persistence()
