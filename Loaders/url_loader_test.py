from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")

docs = loader.load()
content = docs[0].page_content[:500]
print(content)

# needs more preprocessing because of spaces and newlines
