import os
import re
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth, firestore
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import messages_from_dict, messages_to_dict

app = Flask(__name__)
CORS(app)

# Load environment variables from .env
load_dotenv()

# Firebase Configuration
firebase_enabled = False
db = None  # Firestore client
try:
    firebase_credentials = {
        "type": os.getenv("FIREBASE_TYPE"),
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY"),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
        "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
        "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN"),
    }
    if None in firebase_credentials.values():
        raise ValueError("Missing Firebase credential in environment variables")
    
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    firebase_enabled = True
    print("Firebase initialized successfully with Firestore from environment variables")
except Exception as e:
    print(f"Warning: Firebase initialization failed: {e}")
    print("Running without Firebase authentication as fallback")

def verify_firebase_token(id_token):
    if not firebase_enabled:
        print("Firebase disabled, cannot verify token")
        return None
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        print(f"Token verified, UID: {uid}")
        return uid
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
SESSIONS_DIR = Path("user_sessions")
SESSION_EXPIRY_DAYS = 90

SESSIONS_DIR.mkdir(exist_ok=True)

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.3,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=1000,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.1,
    )
    return llm

# Prompt Templates
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
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
db_vector = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
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
    for msg in serialized_messages:
        if "additional_kwargs" not in msg["data"]:
            msg["data"]["additional_kwargs"] = {}
        msg["data"]["additional_kwargs"]["timestamp"] = datetime.now().isoformat()
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
    prefixes = [r'^Assistant:', r'^ANSWER:', r'^YOUR ANSWER:']
    for prefix in prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE).strip()
    text = text.replace('\\n', '\n').strip()
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

    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    if not id_token or not firebase_enabled:
        return jsonify({"error": "Firebase authentication required"}), 401

    user_id = verify_firebase_token(id_token)
    if not user_id:
        return jsonify({"error": "Invalid Firebase ID token"}), 401

    memory = get_session_memory(user_id)
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    prompt = no_docs_prompt.format(chat_history=chat_history, question=user_query)  # Simplified for no-docs case
    response = llm.invoke(prompt)
    cleaned_response = clean_llm_response(response)
    memory.chat_memory.add_user_message(user_query)
    memory.chat_memory.add_ai_message(cleaned_response)
    save_session(user_id, memory)

    return jsonify({
        "response": cleaned_response,
        "user_id": user_id,
        "has_source_documents": False
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

@app.route('/conversation_history', methods=['POST'])
def get_conversation_history():
    data = request.json
    id_token = data.get('idToken')
    target_uid = data.get('target_uid')

    if not id_token or not firebase_enabled:
        return jsonify({"error": "Firebase authentication required"}), 401

    user_id = verify_firebase_token(id_token)
    if not user_id:
        return jsonify({"error": "Invalid Firebase ID token"}), 401

    fetch_uid = user_id
    if target_uid and target_uid != user_id:
        if not is_parent_authorized(user_id, target_uid):
            return jsonify({"error": "Unauthorized to access this user's history"}), 403
        fetch_uid = target_uid

    session_path = get_session_path(fetch_uid)
    if not session_path.exists():
        return jsonify({
            "user_id": fetch_uid,
            "history": [],
            "message": "No conversation history found"
        })
    try:
        with open(session_path, 'r') as f:
            session_data = json.load(f)
        messages = session_data["messages"]
        history = []
        for msg in messages:
            if msg["type"] == "human":
                history.append({
                    "prompt": msg["data"]["content"],
                    "timestamp": msg["data"]["additional_kwargs"].get("timestamp", "N/A")
                })
            elif msg["type"] == "ai":
                if history and "answer" not in history[-1]:
                    history[-1]["answer"] = msg["data"]["content"]
        print(f"Returning history for {fetch_uid}: {history}")
        return jsonify({
            "user_id": fetch_uid,
            "history": history,
            "total_entries": len(history),
            "last_active": session_data["last_active"]
        })
    except Exception as e:
        print(f"Error loading history for user {fetch_uid}: {e}")
        return jsonify({"error": "Failed to load conversation history"}), 500

def is_parent_authorized(parent_uid, child_uid):
    if not db:
        print("Firestore not initialized")
        return False
    try:
        doc_ref = db.collection('parent_child_links').document(parent_uid)
        doc = doc_ref.get()
        if doc.exists:
            child_uids = doc.to_dict().get('child_uids', [])
            return child_uid in child_uids
        return False
    except Exception as e:
        print(f"Error checking authorization: {e}")
        return False

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