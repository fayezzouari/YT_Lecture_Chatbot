from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)

text1 = "foo bar bazzyfoo"

print(text_splitter.split_text(text1)) # TokenTextSplitter splits by tokens

text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

print(text_splitter.split_text(text1)) # TokenTextSplitter splits by chunk
