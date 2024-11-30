from huggingface_hub import InferenceClient

class LLMPipeline:
    def __init__(self, model_name, api_key):
        self.client = InferenceClient(api_key=api_key)
        self.model_name = model_name

    def query(self, messages, max_tokens=500):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message['content']
