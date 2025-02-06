import aisuite as ai
import dotenv
dotenv.load_dotenv()

client = ai.Client()
provider = "groq"
model_id = "llama-3.2-3b-preview"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What’s the weather like in San Francisco?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)