import time
from retriever import load_index
from sentence_transformers import SentenceTransformer
from llm_pipeline import LLMPipeline
from rag_pipeline import RAGPipeline


def load_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def evaluate_rag(rag, questions, ground_truth, top_k=10):
    results = []
    response_times = []

    for i, question in enumerate(questions):
        start_time = time.time()
        response = rag.query(question, top_k=top_k)
        end_time = time.time()
        response_times.append(end_time - start_time)
        
        result = {
            "question": question,
            "response": response,
            "correct": ground_truth[i] in response if ground_truth else None,
            "response_time": response_times[-1]
        }
        results.append(result)
    
    return results, response_times


if __name__ == "__main__":
    index_file = "./index/faiss_index"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    api_key = "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"
    questions_file = "./data/golden_questions.txt"
    ground_truth_file = "./data/golden_answers.txt"

    index = load_index(index_file)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    llm = LLMPipeline(model_name=model_name, api_key=api_key)
    rag = RAGPipeline(index=index, model=model, llm=llm)

    questions = load_questions(questions_file)
    ground_truth = load_questions(ground_truth_file)

    results, response_times = evaluate_rag(rag, questions, ground_truth)

    with open("./data/test_results.txt", 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"Question: {result['question']}\n")
            f.write(f"Response: {result['response']}\n")
            f.write(f"Correct: {result['correct']}\n")
            f.write(f"Response Time: {result['response_time']}s\n")
            f.write("-" * 50 + "\n")

    q50 = sorted(response_times)[len(response_times) // 2]
    q90 = sorted(response_times)[int(len(response_times) * 0.9)]
    q99 = sorted(response_times)[int(len(response_times) * 0.99)]
    precision_at_10 = sum(r['correct'] for r in results if r['correct'] is not None) / len(results)

    print("Performance Metrics:")
    print(f"q50 (Median Response Time): {q50}s")
    print(f"q90 (90% Response Time): {q90}s")
    print(f"q99 (99% Response Time): {q99}s")
    print(f"Precision@10: {precision_at_10 * 100}%")
