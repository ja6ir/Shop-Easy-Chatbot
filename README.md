# ShopEasy Chatbot

This is a customer support chatbot for **ShopEasy** (An Imaginery E-Commerce Platform), designed to assist customers with their queries. It uses **LM Studio** (with Llama-3.2) for natural language processing and **FAISS** (Facebook AI Similarity Search) for fast retrieval of relevant responses from a pre-existing set of labeled training data.

## Features
- **Real-time chat interface** powered by **Streamlit**.
- **Natural language processing** through a large language model (Llama-3.2) running on **LM Studio**.
- **Query retrieval** using **FAISS** index from a set of predefined questions and answers.
- Customizable **chat parameters** such as temperature and token limits.

## Installation

To set up and run this project locally, follow the steps below:

### Prerequisites
- Python 3.x
- `pip` for package management

### Install Dependencies

Clone this repository and install the required libraries:

```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
```

Make sure you have the following installed:
- `streamlit`: For building the web interface.
- `openai`: For interacting with OpenAI's models (if you're using OpenAI API).
- `requests`: To send HTTP requests to the LM Studio API.
- `pandas`, `faiss`, `sentence-transformers`: For data processing, creating the FAISS index, and embeddings.

### Setting up LM Studio
1. Install and set up **LM Studio** locally or on a server.
2. Make sure the LM Studio API is running on the correct URL (default: `http://127.0.0.1:1234`).
3. Ensure that the **Llama-3.2-1B-Instruct** model is properly loaded in LM Studio.

### Setting up FAISS
1. The FAISS index is generated from a CSV of labeled customer queries and responses. If you don’t have a pre-existing file, you can create one following the sample data format below.
2. Use the `index.py` script to generate the FAISS index and save it locally. The script also saves the DataFrame containing queries and responses as a CSV file.

### Running the Chatbot

1. Start the LM Studio API if it's not already running.
2. Run the Streamlit app:

```bash
streamlit run app.py
```

This will launch the chatbot interface in your browser.

## Configuration Options

In the **Sidebar**, you can adjust the following parameters:

- **Temperature**: Controls the randomness of the model's responses. A higher value (e.g., 0.7) means more creativity, while a lower value (e.g., 0.2) makes the model more focused and deterministic.
- **Max Tokens**: Controls the maximum number of tokens the model can generate in a single response. Set to `-1` for unlimited.

## How It Works

1. **Query Input**: The user inputs a query via the Streamlit interface.
2. **Retrieving Relevant Data**: The chatbot uses **FAISS** to retrieve the top 5 most similar queries from a pre-existing database of labeled questions and responses.
3. **Model Response**: The relevant queries are passed to **LM Studio** for generating a response. The Llama model uses the provided context and the input query to generate a helpful reply.
4. **Streaming Response**: The chatbot responds in real-time with a streamed response, updating the conversation history as the model generates it.

### Search Functionality
The `search_vector_store` function encodes the user's query using a **Sentence Transformer** model (`all-MiniLM-L6-v2`) and searches for the most similar queries in the FAISS index. The top `k` results are returned along with their corresponding responses.

### Training Data
The training data consists of a list of labeled customer service queries and responses. Example:

| Query | Response |
|-------|----------|
| "Can I pre-order items that are not yet available?" | "Yes, you can pre-order items. Simply select the item and choose the pre-order option if it's available." |
| "How do I change my account privacy settings?" | "Go to your account settings and select 'Privacy.' From there, you can adjust your privacy preferences." |

The `df` DataFrame contains these queries and responses, which are encoded into embeddings and indexed in **FAISS** for fast retrieval.

### Example Query-Response

**User:** "How do I change my account privacy settings?"  
**Assistant:** "To change your account privacy settings, navigate to your profile settings and select the 'Privacy' tab."

## Contributions

Contributions are welcome! Please feel free to open issues or submit pull requests.

### Example Issue
- **Bug Report**: "The chatbot gives inaccurate responses to privacy-related questions."
- **Feature Request**: "Add support for multi-lingual responses."

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [LM Studio](https://lmstudio.com) for providing the model hosting platform.
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.
- [Sentence Transformers](https://www.sbert.net/) for embedding generation.

## Troubleshooting

If you encounter any issues:

1. Ensure that LM Studio is running and accessible at the correct URL.
2. Check the Streamlit logs for any errors or warnings.
3. Verify that the FAISS index file exists and is properly loaded.

For further assistance, open an issue on GitHub or contact the repository maintainers.

