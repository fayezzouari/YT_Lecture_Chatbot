import os
import dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)
llm = ChatOpenAI(
  openai_api_key=os.getenv("OPENROUTER_API_KEY"),
  openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
  model_name="deepseek/deepseek-r1",
  temperature=0.5,
)
# vectorstore
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=jina_embeddings,  
    persist_directory=persist_directory
)

retriever = vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3})
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,

)

def proof_of_concept():
    chat_history = []

    question = "When did Fayez participate in OSC?"
    result = qa.invoke({"question": question, "chat_history": chat_history})  
    print(result['answer'])

    # Append conversation to history
    chat_history.append({"question": question, "answer": result['answer']})

    question = "What did he do back then?"
    result = qa.invoke({"question": question, "chat_history": chat_history})
    print(result['answer'])

    # Append the second conversation to history
    chat_history.append({"question": question, "answer": result['answer']})

def chatbot():
    chat_history = []

    while True:
        question = str(input("Enter your question: "))
        result = qa.invoke({"question": question, "chat_history": chat_history})
        print(result['answer'])

        # Append conversation to history
        chat_history.append({"question": question, "answer": result['answer']})
        if question == "exit":
            break

if __name__ == "__main__":
    choice = input("Choose: ")
    if choice == "1":
        proof_of_concept()
    else:
        chatbot()