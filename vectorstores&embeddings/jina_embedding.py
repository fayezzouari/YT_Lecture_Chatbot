import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import numpy as np


def embedding(prompt):
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer jina_8c300ed1af414dcfae492bf5d825c65eB1S_170hDLBNvX1c_5sCNpe2qRuy'
    }

    data = {
        "model": "jina-clip-v2",
        "dimensions": 1024,
        "normalized": True,
        "embedding_type": "float",
        "input": [
            prompt
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()['data'][0]['embedding']

emb1=embedding("I am a software engineer")
emb2=embedding("my dad is calling")
emb3=embedding("I am a web developer")


print(np.dot(emb1, emb2))
print(np.dot(emb1, emb3))
print(np.dot(emb2, emb3))

# results:
# 0.3965627940560227
# 0.8351095796073154
# 0.4095282581957695

