from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it")

response = llm.invoke("How many moons does Jupiter have?")
print(response.text)

# call with system prompt and tempratiure

llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it",temperature=1)
response = llm.invoke([
    ["system", "You are a helpful assistant that answers in short one line"],
    ["human", "How many moons does Jupiter have?"]
])
print(response.text)


# call llm using grq api key

from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-20b")

messages = [
    (
        "system",
        "You are a helpful assistant that gives a short answer to a question",
    ),
    (
        "human", 
        "How many moons does Jupiter have?"
    ),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
