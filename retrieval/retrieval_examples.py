from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings  
import requests

persist_directory = 'docs/chroma/'


class JinaAIEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.api_key = api_key

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

jina_embeddings = JinaAIEmbeddings(api_key="jina_8c300ed1af414dcfae492bf5d825c65eB1S_170hDLBNvX1c_5sCNpe2qRuy") # replace with your own API key from environment variables


vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=jina_embeddings
)


print(vectordb._collection.count())

texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]


smalldb = Chroma.from_texts(texts, embedding=jina_embeddings)
question = "Tell me about all-white mushrooms with large fruiting bodies"

print(smalldb.similarity_search(question, k=2))

# provides more diverse results using MMR (maximal marginal relevance) and avoids duplicates
print(smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3))