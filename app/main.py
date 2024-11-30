import streamlit as st
import time
import sys
from pathlib import Path

# Добавляем путь к папке src
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from rag_pipeline import RAGPipeline
from retriever import load_index
from sentence_transformers import SentenceTransformer
from llm_pipeline import LLMPipeline

index_file = "./index/faiss_index"
data_dir = "./data/processed"
llama_model_name = "meta-llama/Llama-3.2-3B-Instruct"
api_key = "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"

index = load_index(index_file)
retriever_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
llm = LLMPipeline(model_name=llama_model_name, api_key=api_key)
rag = RAGPipeline(index=index, model=retriever_model, llm=llm)

st.title("RAG System")
st.write("Enter a query to retrieve relevant documents and get answers.")

user_query = st.text_input("Enter your query:")
top_k = st.slider("Number of retrieved documents:", 1, 30, 5)

if st.button("Submit Query"):
    if user_query.strip() == "":
        st.error("Please enter a query!")
    else:
        st.write("Processing your query...")
        start_time = time.time()
        response = rag.query(user_query, top_k=top_k)
        end_time = time.time()

        st.write("### Response from the model:")
        st.write(response)

        st.write("### Relevant Documents:")
        distances, indices = rag.index.search(retriever_model.encode([user_query]), top_k)
        for i, idx in enumerate(indices[0]):
            st.write(f"Document {i + 1}: {rag.texts[idx][:300]}...")
            st.write(f"Distance: {distances[0][i]}")

        st.write("### Performance Metrics:")
        st.write(f"Response Time: {end_time - start_time:.2f} seconds")
