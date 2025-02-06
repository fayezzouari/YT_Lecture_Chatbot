from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "Your long document with multiple sections and paragraphs..."

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2, chunk_overlap=1, separators=[""]
)

chunks = splitter.split_text(text)
print(chunks)
