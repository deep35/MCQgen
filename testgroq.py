import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

# Initialize ChatGroq normally
from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0.7,
    groq_api_key="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",
    model="mixtral-8x7b-32768"
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
