import config.Config as config
from openai import OpenAI

client = OpenAI(
    api_key = config.apiKey
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "Personal",
            "content": "Say this is a test",
        }
    ],
    model="text-curie-001",
)

print(chat_completion)