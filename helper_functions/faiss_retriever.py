# Step 5: Custom Faiss-based Retriever with Metadata
import numpy as np


def faiss_retriever(index, query_vector, df, k=5):
    query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
    
    # Perform search
    distances, indices = index.search(query_vector, k)
    
    # Filter by score_threshold
    results = []
    for dist, idx in zip(distances[0], indices[0]):   
        result = {
            "sentence_chunk": df.iloc[idx]["sentence_chunk"],
            "page_number": df.iloc[idx]["page_number"],
            "distance": dist
        }
        results.append(result)

    return results
