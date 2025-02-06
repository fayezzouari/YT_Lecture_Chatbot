# YT-Lecture-Chatbot 🎥🤖  

## Repository Purpose 📌  

This repository is dedicated to practicing the skills gained from the **DeepLearning.AI** course:  
[LangChain: Chat with Your Data](https://learn.deeplearning.ai/courses/langchain-chat-with-your-data).  

The main focus of this project is implementing **Retrieval-Augmented Generation (RAG)** to build an intelligent chatbot capable of answering questions based on YouTube lecture transcripts.  

## Topics Covered 🧠  

Throughout this project, I explored the following key concepts:  

✅ **Document Loading** – Extracting subtitles from YouTube videos.  
✅ **Document Splitting** – Breaking subtitles into manageable chunks for processing.  
✅ **Vector Stores & Embeddings** – Converting text into embeddings and storing them using **ChromaDB**.  
✅ **Question Answering** – Building a system to retrieve relevant context and generate responses.  
✅ **Chatbot with Memory** – Enhancing the question-answering system into a conversational chatbot with context retention.  
✅ **Full RAG Implementation** – Integrating retrieval and generation to improve response accuracy.  

## Mini-Project: YT-Lecture-Chatbot 🎬  

As a practical application of these concepts, I developed **YT-Lecture-Chatbot**:  

🔹 Accepts a **YouTube video ID** as input.  
🔹 Extracts **subtitles** from the video.  
🔹 Chunks the text for efficient processing.  
🔹 Generates **embeddings** and stores them in **ChromaDB**.  
🔹 Enables users to chat with the chatbot about the lecture contents.  

## Getting Started 🚀  

1️⃣ Clone the repository:  
```bash
git clone https://github.com/fayezzouari/YT-Lecture-Chatbot.git
cd YT-Lecture-Chatbot
```
2️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```
3️⃣ Run the chatbot:  
```bash
python chatbot.py
```
