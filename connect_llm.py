import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# setup LLM (mistral)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id = huggingface_repo_id,
        temperature = 0.5,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs = {
            "max_length":"512"
        }
    )
    return llm

# connect llm with FAISS and create chain

DB_FAISS_PATH = "vectorstore/db_faiss"
# Modify your CUSTOM_PROMPT_TEMPLATE to allow for conversational responses
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful assistant specializing in drug addiction information.

For any questions about drug addiction, substance abuse treatment, or recovery:
1. ONLY use the factual information provided in the context below
2. Do NOT make up information or provide hotline numbers not mentioned in the context
3. If the context doesn't contain relevant information, say "I don't have specific information about that in my database"

For greetings or casual conversation, respond in a friendly way, but keep responses brief.

Context: {context}
Question: {question}

Answer the question based ONLY on the context provided. Don't invent facts.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
    return prompt

# load database
DB_FAISS_PATH = "vectorstore/db_faiss" 
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(HUGGINGFACE_REPO_ID),
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs = {'k':3, 'score_threshold': 0.3}),
    return_source_documents = True,
    chain_type_kwargs = {'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# invoke with a query
user_query = input("Write query here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
print("Number of documents in FAISS:", db.index.ntotal)
