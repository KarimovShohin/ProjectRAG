
# RAG (Retrieval-Augmented Generation) Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline for answering user queries by combining retrieval of relevant context and response generation using a large language model (LLM).

## Project Structure

```
rag_project/
├── data/
│   ├── raw/
│   │   ├── 1.fb2
│   │   ├── 2.fb2
│   │   ├── 3.fb2
│   │   ├── 4.fb2
│   │   ├── 5.fb2
│   │   ├── 6.fb2
│   ├── processed/
│   │   ├── book_1.txt
│   │   ├── book_2.txt
│   │   ├── book_3.txt
│   │   ├── book_4.txt
│   │   ├── book_5.txt
│   │   ├── book_6.txt
├── src/
│   ├── preprocess.py
│   ├── retriever.py
│   ├── llm_pipeline.py
│   ├── rag_pipeline.py
│   ├── evaluate.py
├── app/
│   ├── main.py
├── requirements.txt
├── README.md
```

## Key Features

- **Data Preparation**: Preprocesses raw text data into a structured format for indexing.
- **Retriever**: Uses Sentence Transformers to encode and index the data, enabling efficient retrieval of relevant documents.
- **LLM Integration**: Queries a large language model (LLM) from Hugging Face's Inference API for generating answers based on retrieved context.
- **Evaluation**: Supports testing and evaluation with a golden set of questions and answers.
- **Web Interface**: Provides a Streamlit-based web interface for user interaction.


## Usage

### Preprocessing Data
Run the `preprocess.py` script to process raw data:
```
python src/preprocess.py
```

### Building the Index
Run the `retriever.py` script to create the FAISS index:
```
python src/retriever.py
```

### Testing and Evaluation
Run the `evaluate.py` script to evaluate the RAG pipeline with the golden set:
```
python src/evaluate.py
```

### Running the Web Interface
Start the Streamlit app:
```
streamlit run app/main.py
```

## Dependencies

- Python 3.9 or higher
- Sentence Transformers
- Hugging Face Transformers
- FAISS
- Streamlit

## Performance Metrics

- **q50 (Median Response Time)**: ~2.76 seconds
- **q90 (90% Response Time)**: ~8.61 seconds
- **q99 (99% Response Time)**: ~9.14 seconds
- **Precision@10**: 0.0% 
