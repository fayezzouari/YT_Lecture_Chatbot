import os
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings  
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader

import dotenv

dotenv.load_dotenv()

persist_directory = 'docs/chroma/'


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
jina_api_key = "jina_8c300ed1af414dcfae492bf5d825c65eB1S_170hDLBNvX1c_5sCNpe2qRuy"
jina_embeddings = JinaAIEmbeddings(api_key = jina_api_key) # replace with your own API key from environment variables

# loader
loader = PyPDFLoader("Loaders\\docs\\resume.pdf")
docs = loader.load()

# text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)
splits = text_splitter.split_documents(docs)

persist_directory = "docs/chroma/"

# vectorstore
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=jina_embeddings,  
    persist_directory=persist_directory
)



question = "what did Fayez do this summer?"
docs = vectordb.similarity_search(question,k=3)


llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0)


qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff", 

    retriever=vectordb.as_retriever()
)
print(qa_chain)
result = qa_chain.invoke({"query": question})

print(result)