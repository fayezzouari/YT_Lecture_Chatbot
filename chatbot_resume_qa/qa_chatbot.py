import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

import dotenv

from jina_embeddings import JinaAIEmbeddings

dotenv.load_dotenv()



jina_api_key = os.getenv("JINA_API_KEY")
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



question = str(input("Enter your question: "))
docs = vectordb.similarity_search(question,k=3)

llm = ChatOpenAI(
  openai_api_key=os.getenv("OPENROUTER_API_KEY"),
  openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
  model_name="google/gemini-2.0-flash-001",
  temperature=0.5,
)


qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff", 

    retriever=vectordb.as_retriever()
)
result = qa_chain.invoke({"query": question})

print(result['result'])