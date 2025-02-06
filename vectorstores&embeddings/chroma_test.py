import requests
import json
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.base import Embeddings  

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

loader = PyPDFLoader("Loaders\\docs\\resume.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)
splits = text_splitter.split_documents(docs)

persist_directory = "docs/chroma/"

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=jina_embeddings,  
    persist_directory=persist_directory
)

print(vectordb._collection.count())

question = "what did fayez do in 2024?"
docs = vectordb.similarity_search(question,k=3)

print(docs[0].page_content)

vectordb.persist() # save the vectorstore to disk