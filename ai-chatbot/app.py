import os
import re
import json
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import messages_from_dict, messages_to_dict

app = Flask(__name__)
CORS(app)

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Upgrade to a more capable model
SESSIONS_DIR = Path("user_sessions")
SESSION_EXPIRY_DAYS = 90
FIREBASE_CREDENTIALS_PATH = os.environ.get("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")

SESSIONS_DIR.mkdir(exist_ok=True)

# Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)
    firebase_enabled = True
    print("Firebase authentication enabled")
except Exception as e:
    firebase_enabled = False
    print(f"Warning: Firebase initialization failed: {e}")
    print("Running without Firebase authentication")

def verify_firebase_token(id_token):
    if not firebase_enabled:
        return None
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token['uid']
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.3,  # Lower for more focused responses
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=1000,  # Increase for detailed answers
        top_k=40,  # Slightly reduce for better relevance
        top_p=0.95,  # Higher for more coherent sampling
        repetition_penalty=1.1,  # Prevent repetition
    )
    return llm

# Enhanced Prompt Templates
CUSTOM_PROMPT_TEMPLATE = """
You are a friendly, knowledgeable assistant specializing in drug addiction prevention for youth. Your goal is to provide clear, concise, and actionable advice tailored to young people.

PREVIOUS CONVERSATION:
{chat_history}

INSTRUCTIONS:
1. Use the CONTEXT below and PREVIOUS CONVERSATION to answer the QUESTION accurately.
2. Provide practical, youth-focused advice when possible (e.g., steps, examples, or resources).
3. If the CONTEXT lacks specific details, use general knowledge about drug prevention but stay factual.
4. Keep responses engaging, concise (under 300 words), and free of jargon.
5. Maintain continuity with past exchanges—reference prior facts or questions naturally.

CONTEXT: {context}

QUESTION: {question}

ANSWER:
"""

NO_DOCS_PROMPT = """
You are a friendly, knowledgeable assistant focused on drug addiction prevention for youth. No specific context is available, so rely on general knowledge and past conversation.

PREVIOUS CONVERSATION:
{chat_history}

INSTRUCTIONS:
1. Answer the QUESTION with clear, actionable, youth-oriented advice based on general knowledge.
2. Stay factual—don’t invent details or scenarios beyond what’s in the PREVIOUS CONVERSATION.
3. Keep responses concise (under 300 words), engaging, and relevant to young people.
4. If unsure, say: "I don’t have enough info to give a detailed answer, but here’s what I can share."
5. Reference prior conversation naturally to maintain continuity.

QUESTION: {question}

ANSWER:
"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "chat_history", "question"])

def set_no_docs_prompt():
    return PromptTemplate(template=NO_DOCS_PROMPT, input_variables=["chat_history", "question"])

# Load database and LLM
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")  # Upgrade embedding model
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
llm = load_llm(HUGGINGFACE_REPO_ID)

qa_prompt = set_custom_prompt()
no_docs_prompt = set_no_docs_prompt()

# In-memory cache for active sessions
active_sessions = {}

def get_session_path(user_id):
    return SESSIONS_DIR / f"{user_id}.json"

def save_session(user_id, memory):
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    serialized_messages = messages_to_dict(chat_history)
    session_data = {
        "last_active": datetime.now().isoformat(),
        "messages": serialized_messages
    }
    with open(get_session_path(user_id), 'w') as f:
        json.dump(session_data, f)

def load_session(user_id):
    session_path = get_session_path(user_id)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    if session_path.exists():
        try:
            with open(session_path, 'r') as f:
                session_data = json.load(f)
            session_data["last_active"] = datetime.now().isoformat()
            messages = messages_from_dict(session_data["messages"])
            for message in messages:
                memory.chat_memory.add_message(message)
            with open(session_path, 'w') as f:
                json.dump(session_data, f)
            print(f"Loaded session for user: {user_id}")
        except Exception as e:
            print(f"Error loading session for user {user_id}: {e}")
    else:
        print(f"Created new session for user: {user_id}")
    return memory

def get_session_memory(user_id):
    if user_id in active_sessions:
        print(f"Using cached session for user: {user_id}")
        return active_sessions[user_id]
    memory = load_session(user_id)
    active_sessions[user_id] = memory
    return memory

def clean_llm_response(text):
    """Clean and format the LLM response."""
    # Remove unwanted prefixes
    prefixes = [r'^Assistant:', r'^ANSWER:', r'^YOUR ANSWER:']
    for prefix in prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE).strip()
    
    # Replace escaped newlines with actual newlines and clean up
    text = text.replace('\\n', '\n').strip()
    
    # Ensure the response ends with a period if it’s a sentence
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text

def cleanup_expired_sessions():
    now = datetime.now()
    count = 0
    for session_file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            last_active = datetime.fromisoformat(session_data["last_active"])
            if (now - last_active).days > SESSION_EXPIRY_DAYS:
                session_file.unlink()
                user_id = session_file.stem
                if user_id in active_sessions:
                    del active_sessions[user_id]
                count += 1
        except Exception as e:
            print(f"Error cleaning up session {session_file}: {e}")
    return count

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    id_token = data.get('idToken')
    session_id = data.get('session_id', 'anonymous')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Verify Firebase token
    user_id = None
    if id_token and firebase_enabled:
        user_id = verify_firebase_token(id_token)
        if not user_id:
            return jsonify({"error": "Invalid Firebase ID token"}), 401
        session_id = user_id
    else:
        user_id = session_id

    # Get user memory
    memory = get_session_memory(user_id)

    # Create QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={'k': 5, 'score_threshold': 0.7}),  # Increase k, stricter threshold
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        get_chat_history=lambda h: h,
        verbose=True
    )

    # Get response
    try:
        chain_response = qa_chain.invoke({'question': user_query})
        source_docs = chain_response.get("source_documents", [])
        current_answer = chain_response.get("answer", "I’m not sure how to respond to that.")
        cleaned_answer = clean_llm_response(current_answer)

        # Fallback if no relevant documents
        if not source_docs or len(cleaned_answer) < 50:  # If answer is too short, assume it’s weak
            print(f"No/weak documents for user {user_id}, using fallback prompt")
            chat_history = memory.load_memory_variables({})["chat_history"]
            general_response = llm.invoke(no_docs_prompt.format(
                chat_history=chat_history,
                question=user_query
            ))
            cleaned_answer = clean_llm_response(general_response)
    except Exception as e:
        print(f"Error in QA chain for user {user_id}: {e}")
        cleaned_answer = "Sorry, I hit a snag. Try asking again!"

    # Save session
    save_session(user_id, memory)

    print(f"USER: {user_id}")
    print("CLEANED RESULT: ", cleaned_answer)
    print("SOURCE DOCUMENTS COUNT: ", len(source_docs))

    return jsonify({
        "response": cleaned_answer,
        "has_source_documents": len(source_docs) > 0,
        "user_id": user_id
    })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    data = request.json
    id_token = data.get('idToken')
    fallback_session_id = data.get('session_id', 'anonymous')

    user_id = None
    if id_token and firebase_enabled:
        user_id = verify_firebase_token(id_token)
        if not user_id:
            return jsonify({"error": "Invalid Firebase ID token"}), 401
    else:
        user_id = fallback_session_id

    if user_id in active_sessions:
        del active_sessions[user_id]
    
    session_path = get_session_path(user_id)
    if session_path.exists():
        session_path.unlink()
        return jsonify({"message": f"History for user {user_id} cleared successfully"})
    
    return jsonify({"message": f"No history found for user {user_id}"})

@app.route('/admin/active_users', methods=['GET'])
def list_active_users():
    if not request.headers.get('X-Admin-Key') == os.environ.get('ADMIN_API_KEY', 'admin-secret'):
        return jsonify({"error": "Unauthorized"}), 401
    disk_users = [f.stem for f in SESSIONS_DIR.glob("*.json")]
    return jsonify({
        "in_memory_users": list(active_sessions.keys()),
        "all_users": disk_users,
        "memory_count": len(active_sessions),
        "total_count": len(disk_users)
    })

@app.route('/admin/cleanup_sessions', methods=['POST'])
def cleanup_sessions():
    if not request.headers.get('X-Admin-Key') == os.environ.get('ADMIN_API_KEY', 'admin-secret'):
        return jsonify({"error": "Unauthorized"}), 401
    deleted_count = cleanup_expired_sessions()
    return jsonify({"message": f"Cleaned up {deleted_count} expired sessions"})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "firebase_enabled": firebase_enabled})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)