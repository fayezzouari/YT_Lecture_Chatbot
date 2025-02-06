import requests
from langchain.embeddings.base import Embeddings  


class JinaAIEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.api_key: str = api_key

    def embed_documents(self, texts):
        """Embeds multiple texts at once (for Chroma compatibility)"""
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            "model": "jina-clip-v2",
            "dimensions": 1024, 
            "normalized": True,
            "embedding_type": "float",
            "input": texts  
        }
        response = requests.post(url, headers=headers, json=data)
        return [item['embedding'] for item in response.json()['data']]

    def embed_query(self, text):
        """Embeds a single query text"""
        return self.embed_documents([text])[0]