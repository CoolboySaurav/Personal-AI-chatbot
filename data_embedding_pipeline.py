import os
import google.generativeai as genai
from spacy.lang.en import English
import faiss
import numpy as np
import pandas as pd
import re
import json
from dotenv import load_dotenv
from helper_functions.open_read_pdf import open_read_pdf
from helper_functions.split_list import split_list

# Load the environment variables
load_dotenv()
path = os.getcwd()
# Loading all the data files 
pdf_path1 = path + "/data/saurav.pdf"
pdf_path2 = path + "/data/short_cv.pdf"
faiss_index = "data/faiss_index.bin"


google_api_key = os.getenv("GOOGLE_API_KEY")
print(google_api_key)
genai.configure(api_key=google_api_key)

# Open and read the PDF
pages_and_text = open_read_pdf(pdf_path1)
# pages_and_text2 = open_read_pdf(pdf_path2)
# Remove the last page as it is empty
#pages_and_text1.pop(-1)

# # Combine the two lists of pages and text
# pages_and_text = pages_and_text1 + pages_and_text2

nlp = English()
nlp.add_pipe("sentencizer")

for item in pages_and_text:
    # Add sentences to each page
    item["sentences"] = list(nlp(item["text"]).sents)
    # Make sure all sentences are strings
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    # Count the sentences on each page  
    item["page_sentence_count_spacy"] = len(item["sentences"])

# Loop through pages and texts and split sentences into chunks
num_sentence_chunk_size = 10 
for item in pages_and_text:
    item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                         slice_size=num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])

# Split each chunk into its own item
pages_and_chunks = []
for item in pages_and_text:
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]
        
        # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo 
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters
        
        pages_and_chunks.append(chunk_dict)

# Embed each chunk
for item in pages_and_chunks:
    item["embedding"] = genai.embed_content(
        model="models/text-embedding-004",
        content=item["sentence_chunk"],
        task_type="retrieval_document",
        title="profile"
    )

# Now store 'page_number', 'sentence_chunk', 'chunk_word_count', 'chunk_token_count', 'embedding'['embedding'] into new dictionary

embeddings = []
for item in pages_and_chunks:
    embeddings.append({
        "page_number": item["page_number"],
        "sentence_chunk": item["sentence_chunk"],
        "chunk_word_count": item["chunk_word_count"],
        "chunk_token_count": item["chunk_token_count"],
        "embedding": item["embedding"]["embedding"]
    })

# Store pages_and_chunks into a json file
with open('data/embeddings.json', 'w') as f:
    json.dump(embeddings, f)

df = pd.DataFrame(embeddings)
# Save the DataFrame to a CSV file for later use
df.to_csv("data/embeddings.csv", index=False)

# Convert the embeddings to numpy arrays
df['embedding'] = df['embedding'].apply(np.array)

# Convert the embeddings to numpy arrays and float32 type for faiss indexing 
text_embeddings = np.array(df['embedding'].to_list()).astype('float32')

# Initialize FAISS Index (using IndexFlatL2)
embedding_dim = text_embeddings.shape[1] # Dimensionality of the embeddings
index = faiss.IndexFlatL2(embedding_dim) # L2 distance is the Euclidean distance

# Add the embeddings to the index
index.add(text_embeddings)

# SAVE THE FAISS INDEX
faiss.write_index(index, faiss_index)

