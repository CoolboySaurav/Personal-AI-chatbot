
# Personal AI Chatbot

## Overview
This project is a chatbot application that utilizes the Gemini API for generating responses and FAISS for storing and retrieving document embeddings. It integrates various document types, such as PDF and CSV, to enhance the chatbot's knowledge base and improve its response accuracy.

## Features
- **Document Embedding**: Utilizes the Gemini API to generate embeddings for text data.
- **Similarity Matching**: Implements FAISS for efficient similarity searches among embedded documents.
- **Support for Multiple Document Types**: Loads data from both PDF and CSV formats.
- **In-Memory Document Storage**: Stores original documents for retrieval alongside their embeddings.

## Project Structure
```plaintext
project-directory/
│
├── data/
│   ├── about_me.csv         # Sample CSV file containing user data
│   └── resume.pdf           # Sample PDF file containing resume
│
├── src/
│   ├── chatbot.py           # Main chatbot logic
│   ├── embeddings.py        # Code for generating embeddings
│   └── document_loaders.py  # Code for loading documents
│
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd project-directory
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have access to the Gemini API and set up any necessary authentication.

## Usage

1. Prepare your data files in the `data/` directory (CSV and PDF files).
2. Run the chatbot application:
    ```bash
    python src/chatbot.py
    ```

3. Interact with the chatbot in the command line or through a web interface (if implemented).

## Code Explanation

### Embedding Function
The `embed_content` function generates embeddings for documents using the Gemini API.

```python
def embed_content(data):
     content_data = [str(doc.page_content) for doc in data]
     result = genai.embed_content(
          model="models/text-embedding-004",
          content=content_data,
          task_type="retrieval_document",
          title="profile"
     )
     return result['data']
```

### FAISS Integration
FAISS is used to store and manage document embeddings for efficient retrieval.

```python
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

docstore = InMemoryDocstore(dict(enumerate(documents)))
faiss_store = FAISS(index=index, docstore=docstore, index_to_docstore_id={i: i for i in range(len(documents))})
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you wish to add.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [Gemini API](https://gemini.example.com) for embedding services.
- [FAISS](https://faiss.ai) for fast similarity search and clustering of dense vectors.

### Instructions for Customization:
- Replace `<repository-url>` with the actual URL of your project repository.
- Update any specific details related to your project structure, especially under **Project Structure**.
- If you have specific installation instructions or dependencies, add those in the **Installation** section.
- Adjust any code snippets to match your actual implementation.

Feel free to modify any part of the README to better fit your project or personal style!
