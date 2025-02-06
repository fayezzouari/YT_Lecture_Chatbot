from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
import google.generativeai as genai
dotenv.load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
genai.configure(api_key="AIzaSyD8cKZVou4O2B2uEQEqvH1pEG6l51quwI8")

models = genai.list_models()
for model in models:
    print(f"Available model: {model.name}")
# Test the model
"""response = llm.invoke("Hello, how are you?")
print(response)"""