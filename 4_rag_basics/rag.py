import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
load_dotenv() 
assert os.getenv("GROQ_API_KEY"), "Set GROQ_API_KEY in a .env file or above"
print("API key loaded.")

# load the pdf
PDF_PATH="telecom_guide.pdf"
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"Loaded {len(pages)} pages from the PDF.")
print("\n--- First page preview (first 500 chars) ---")
print(pages[0])


from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,       # ~150 words per chunk
    chunk_overlap=100,    # overlap keeps context at boundaries
    separators=["\n\n", "\n", ".", " "],  # tries paragraph → line → sentence → word
)

chunks = splitter.split_documents(pages)

print(f"Total chunks: {len(chunks)}  (from {len(pages)} pages)")
print(f"Avg chunk length: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
print("\n--- Example chunk ---")
print(chunks[5].page_content)


# embeded chunks and store in chroma db

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Loading embedding model (downloads ~90 MB on first run)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Embedding all chunks and storing in Chroma (in memory)...")
vector_store = Chroma.from_documents(chunks, embeddings)

print(f"Vector store ready. {vector_store._collection.count()} vectors stored.")


# reteriver

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

test_query = "What is VoLTE and how does it improve call quality?"
retrieved = retriever.invoke(test_query)

print(f"Query: {test_query}")
print(f"Retrieved {len(retrieved)} chunks:\n")
for i, doc in enumerate(retrieved, 1):
    print(f"--- Chunk {i} ---")
    print(doc.page_content[:300])
    print()


# build the rag chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq


# --- Helper: join retrieved chunks into a single context string ---
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# --- System prompt: ground the LLM in the retrieved context ---
SYSTEM_PROMPT = """\
You are a helpful telecom assistant.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

# --- LLM via Groq API ---
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    reasoning_format="parsed",
)

# --- Assemble the chain ---
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG chain assembled.")

# ask a single  question
question = "How does international roaming work and what charges should I expect?"

print(f"Q: {question}\n")
print("A:", chain.invoke(question))

# intractive Q&A loop

print("Telecom RAG Assistant — type 'quit' to exit\n")

while True:
    question = input("Your question: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    if not question:
        continue

    print("\nAnswer:")
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")