def create_unique_chunk_id(video_name, chunk_number):
    """Create a unique ID for a chunk by combining video name and chunk number"""
    # Clean video name to create a valid identifier
    video_slug = video_name.lower().replace(' ', '_').replace('|', '').replace('-', '_')
    return f"{video_slug}_chunk_{chunk_number}"

def create_evaluation_dataset():
    """
    Creates a dataset of test queries and their ground truth relevant chunks
    based on the 'Deep Learning in Tamil Part 5' summary.
    """
    video_name = "Deep Learning in Tamil | Chain Rule in Back Propagation | Deep Learning for Beginners | Part 5.mp4"

    test_queries = [
        "What is the main topic of this video?",
        "What is the formula for the weight update rule in gradient descent?",
        "What is the Sum of Squared Errors loss function?",
        "How is the chain rule used to update weights in a neural network?",
        "How do you calculate the gradient for a weight in the final layer?",
        "What is the difference between updating weights in the final layer versus an earlier hidden layer?",
        "How does a weight in a hidden layer influence the final loss?",
        "What does it mean to 'backpropagate' the error?",
        "Can you show the full chain rule formula for calculating the gradient of a weight in a hidden layer?",
        "What is the overall goal of training a neural network as described in the video?"
    ]
    
    # Each query maps to video name + chunk number combinations
    relevant_chunks = {
        "What is the main topic of this video?": [
            (video_name, 1),
            (video_name, 3),
            (video_name, 10),
        ],
        "What is the formula for the weight update rule in gradient descent?": [
            (video_name, 2),
            (video_name, 3),
            (video_name, 5),
            (video_name, 6),
        ],
        "What is the Sum of Squared Errors loss function?": [
            (video_name, 2),
            (video_name, 5),
            (video_name, 8),
        ],
        "How is the chain rule used to update weights in a neural network?": [
            (video_name, 3),
            (video_name, 4),
            (video_name, 5),
            (video_name, 6),
        ],
        "How do you calculate the gradient for a weight in the final layer?": [
            (video_name, 3),
            (video_name, 4),
            (video_name, 6),
        ],
        "What is the difference between updating weights in the final layer versus an earlier hidden layer?": [
            (video_name, 7),
            (video_name, 8),
        ],
        "How does a weight in a hidden layer influence the final loss?": [
            (video_name, 7),
            (video_name, 8),
            (video_name, 9),
        ],
        "What does it mean to 'backpropagate' the error?": [
            (video_name, 8),
            (video_name, 9),
            (video_name, 10),
        ],
        "Can you show the full chain rule formula for calculating the gradient of a weight in a hidden layer?": [
            (video_name, 9),
        ],
        "What is the overall goal of training a neural network as described in the video?": [
            (video_name, 2),
        ]
    }
    
    # Convert to unique IDs
    relevant_ids = {}
    for query, chunks in relevant_chunks.items():
        relevant_ids[query] = [create_unique_chunk_id(video, chunk_num) for video, chunk_num in chunks]
    
    return test_queries, relevant_ids

def evaluate_rag_pipeline(embedder, test_queries, relevant_ids, collection_name="video_summaries", search_method="bm25_mmr"):
    results = []
    
    for query in test_queries:
        # Get the relevant chunk IDs for this query
        true_relevant = relevant_ids[query]
        
        # Run the query through your RAG system
        search_results = embedder.search_videos(query, collection_name=collection_name, n_results=5, search_method=search_method)
        
        # Extract unique IDs from the returned results (PREVENT DUPLICATES)
        retrieved_ids = []
        seen_ids = set()
        
        for metadata in search_results["metadatas"]:
            video_name = metadata["video_name"]
            chunk_number = metadata["chunk_number"]
            unique_id = create_unique_chunk_id(video_name, chunk_number)
            
            # Only add if we haven't seen this ID yet
            if unique_id not in seen_ids:
                retrieved_ids.append(unique_id)
                seen_ids.add(unique_id)
        
        # Calculate metrics (with safeguard)
        relevant_retrieved = [id for id in retrieved_ids if id in true_relevant]
        
        # Ensure we don't count more relevant items than exist
        if len(relevant_retrieved) > len(true_relevant):
            print(f"WARNING: Found more relevant items than exist for query: {query}")
            print(f"Retrieved: {relevant_retrieved}")
            print(f"True relevant: {true_relevant}")
            relevant_retrieved = relevant_retrieved[:len(true_relevant)]
        
        precision_at_5 = len(relevant_retrieved) / min(5, len(retrieved_ids))
        recall_at_5 = len(relevant_retrieved) / len(true_relevant)
        
        results.append({
            "query": query,
            "precision@5": precision_at_5,
            "recall@5": recall_at_5,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": true_relevant,
            "matched_ids": relevant_retrieved
        })
    # Calculate average metrics
    avg_precision = sum(r["precision@5"] for r in results) / len(results)
    avg_recall = sum(r["recall@5"] for r in results) / len(results)
    
    # Return the complete evaluation results
    return {
        "detailed_results": results,
        "average_precision@5": avg_precision,
        "average_recall@5": avg_recall
    }

if __name__ == "__main__":
    from rag_pipeline.video_embedder import VideoEmbedder
    
    # Initialize the embedder
    embedder = VideoEmbedder()
    
    # Create the evaluation dataset
    test_queries, relevant_ids = create_evaluation_dataset()
    

    # Evaluate the RAG pipeline
    search_methods = ["similarity", "mmr", "hybrid", "bm25_mmr"]
    results = {}
    for method in search_methods:
        print(f"Evaluating with search method: {method}")
        evaluation_results = evaluate_rag_pipeline(embedder, test_queries, relevant_ids, search_method=method)
        # Print the results
        print("Evaluation Results:")
        # for result in evaluation_results["detailed_results"]:
        #     print(f"Query: {result['query']}")
        #     print(f"Precision@5: {result['precision@5']:.2f}, Recall@5: {result['recall@5']:.2f}")
        #     print(f"Retrieved IDs: {result['retrieved_ids']}")
        #     print(f"Relevant IDs: {result['relevant_ids']}")
        #     print("-" * 40)
        
        print(f"Average Precision@5: {evaluation_results['average_precision@5']:.2f}")
        print(f"Average Recall@5: {evaluation_results['average_recall@5']:.2f}")
        results[method] = {
            "average_precision@5": evaluation_results["average_precision@5"],
            "average_recall@5": evaluation_results["average_recall@5"]
        }
    print("Final Results Summary:")
    for method, metrics in results.items():
        print(f"Method: {method}, Average Precision@5: {metrics['average_precision@5']:.2f}, Average Recall@5: {metrics['average_recall@5']:.2f}")