from sentence_transformers import SentenceTransformer
from retriever import load_index, search_index, load_data
from llm_pipeline import LLMPipeline

class RAGPipeline:
    def __init__(self, index, model, llm):
        self.index = index
        self.model = model
        self.llm = llm
        self.data_dir = "./data/processed"
        self.texts = load_data(self.data_dir)

    def query(self, user_query, top_k=5, max_context_length=3000):
        distances, indices = search_index(self.index, self.model, user_query, top_k)
        retrieved_texts = [self.texts[idx] for idx in indices[0]]
        context = " ".join(retrieved_texts)
        if len(context) > max_context_length:
            context = context[:max_context_length]
        messages = [
            {"role": "system", "content": "Use the provided context to answer the user's question."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_query}"}
        ]
        response = self.llm.query(messages)
        return response

if __name__ == "__main__":
    index_file = "./index/faiss_index"
    index = load_index(index_file)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    llama_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    api_key = "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"
    llm = LLMPipeline(model_name=llama_model_name, api_key=api_key)
    rag = RAGPipeline(index=index, model=model, llm=llm)
    user_query = "Who are Harry Potter?"
    response = rag.query(user_query)
    print("Response from LLM:")
    print(response)
