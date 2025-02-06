from langchain_community.document_loaders import PyPDFLoader

# load the pdf
loader = PyPDFLoader("Loaders\\docs\\resume.pdf")
pages = loader.load()

# print number of pages in pdf
print(len(pages))

# print first page of pdf
print(pages[0])

# print metadata of first page
print(pages[0].metadata)