import os
import dotenv
import gradio as gr
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from jina_embeddings import JinaAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
import re


dotenv.load_dotenv()

# Load API keys
jina_api_key = os.getenv("JINA_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")

# Initialize embeddings
jina_embeddings = JinaAIEmbeddings(api_key=jina_api_key)


# Function to get subtitles
def get_subtitles(video_id, language='en'):
    res=""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        for line in transcript:
           
            res+=line["text"]+ " "
        return res
    except Exception as e:
        print(f"Could not retrieve subtitles: {e}")

def process_transcript(transcript_path):
    global qa  
    text=get_subtitles(transcript_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    splits = text_splitter.split_text(text)

    persist_directory = "docs/chroma/"
    vectordb = Chroma.from_texts(
        splits,
        embedding=jina_embeddings,
        persist_directory=persist_directory
    )

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = ChatOpenAI(
        openai_api_key=openrouter_api_key,
        openai_api_base=openrouter_base_url,
        model_name="nvidia/llama-3.1-nemotron-70b-instruct:free",
        temperature=0.5,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )

    return "Transcript processed successfully! You can now ask questions.", gr.update(visible=True)

chat_history = []

def ask_question(question):
    global qa

    if not qa:
        return "⚠️ **Please upload and process a transcript first.**"
    
    result = qa.invoke({"question": question, "chat_history": chat_history})
    chat_history.append({"question": question, "answer": result['answer']})

    return result['answer']

# Create Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# 🎥 YouTube Transcript QA Chatbot")
    chatbot_section = gr.Column(visible=False)  # Initially hidden

    with gr.Row():
        transcript_file = gr.Textbox(label="Upload YouTube Transcript (TXT)")
        process_button = gr.Button("Process Transcript")

    status_output = gr.Markdown()  # Markdown for status message

    chatbot_section = gr.Column(visible=False)  # Initially hidden
    with chatbot_section:
        gr.Markdown("### 🤖 Ask Questions about the Video")
        question_input = gr.Textbox(label="Your Question")
        answer_output = gr.Markdown()  # Display results in Markdown
        ask_button = gr.Button("Ask")

    process_button.click(
        process_transcript,
        inputs=transcript_file,
        outputs=[status_output, chatbot_section]
    )

    ask_button.click(ask_question, inputs=question_input, outputs=answer_output)

# Run the Gradio app
if __name__ == "__main__":
    interface.launch()
