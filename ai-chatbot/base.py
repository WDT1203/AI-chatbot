import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)
CORS(app)

# setup LLM (mistral)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id = huggingface_repo_id,
        temperature = 0.6,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=300,  
        top_k=50,
        top_p=0.9,
        model_kwargs = {
            "max_length": 512,
        }
    )
    return llm

# More explicit prompt template with stricter instructions
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful, friendly assistant that provides accurate information about drug addiction prevention for youth.

PREVIOUS CONVERSATION:
{chat_history}

INSTRUCTIONS:
1. Use the CONTEXT provided below AND the PREVIOUS CONVERSATION to answer the question.
2. If the CONTEXT doesn't contain relevant information, you can use the conversation history and general knowledge.
3. NEVER make up information or include imaginary conversations.
4. Maintain continuity with previous exchanges - remember facts, people, and stories mentioned earlier.
5. DO NOT include phrases like "According to the context" in your response.

CONTEXT: {context}

CURRENT QUESTION: {question}

YOUR ANSWER:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "chat_history", "question"])
    return prompt

# Conditional prompt when no relevant documents are found
NO_DOCS_PROMPT = """
You are a helpful, friendly assistant specializing in drug addiction prevention for youth.

PREVIOUS CONVERSATION:
{chat_history}

INSTRUCTIONS:
1. This is a question without specific context from the knowledge base.
2. Use the PREVIOUS CONVERSATION and general knowledge to provide a helpful response.
3. If unsure, politely state that you don't have enough information to provide a reliable answer.
4. Maintain continuity with previous exchanges - remember facts, people, and stories mentioned earlier.
5. NEVER make up information beyond what was previously discussed.

QUESTION: {question}

YOUR ANSWER:
"""

def set_no_docs_prompt():
    prompt = PromptTemplate(template=NO_DOCS_PROMPT, input_variables=["chat_history", "question"])
    return prompt

# Load database
DB_FAISS_PATH = "vectorstore/db_faiss" 
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Define memory with explicit input and output keys
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    return_messages=True
)

# Create custom prompts
qa_prompt = set_custom_prompt()
no_docs_prompt = set_no_docs_prompt()

# Load the LLM once
llm = load_llm(HUGGINGFACE_REPO_ID)

# Create a conversational chain with memory AND custom prompt
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={'k': 3, 'score_threshold': 0.5}),  # Reduced k for more focused results
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    get_chat_history=lambda h: h,
    verbose=True
)

def clean_llm_response(text):
    """Clean the LLM response but preserve content."""
    prefixes = [r'^Assistant:', r'^YOUR ANSWER:']
    for prefix in prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE).strip()
    
    return text

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    session_id = data.get('session_id', 'default')  # Get session ID
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Get the response from the chain
    chain_response = qa_chain.invoke({'question': user_query})
    
    # Get source documents
    source_docs = chain_response.get("source_documents", [])
    
    # Extract and clean the current answer
    current_answer = chain_response.get("answer", "I'm not sure how to respond to that.")
    cleaned_answer = clean_llm_response(current_answer)
    
    # If no source documents were found, use the general knowledge prompt
    if not source_docs:
        print("No relevant documents found, using general knowledge prompt with chat history")
        chat_history = memory.load_memory_variables({})["chat_history"]
        general_response = llm.invoke(no_docs_prompt.format(
            chat_history=chat_history,
            question=user_query
        ))
        cleaned_answer = clean_llm_response(general_response)
    print("ORIGINAL RESULT: ", current_answer)
    print("CLEANED RESULT: ", cleaned_answer)
    print("SOURCE DOCUMENTS COUNT: ", len(source_docs))
    print("Number of documents in FAISS:", db.index.ntotal)

    # Return only the cleaned current answer
    return jsonify({
        "response": cleaned_answer,
        "has_source_documents": len(source_docs) > 0
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)