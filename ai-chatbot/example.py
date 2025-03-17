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
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
SESSIONS_DIR = Path("user_sessions")
SESSION_EXPIRY_DAYS = 90  # Longer expiry for authenticated users
FIREBASE_CREDENTIALS_PATH = os.environ.get("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")

# Create sessions directory if it doesn't exist
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
    """Verify the Firebase ID token and return the user ID."""
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
        temperature=0.6,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=300,
        top_k=50,
        top_p=0.9,
        model_kwargs={
            "max_length": 512,
        }
    )
    return llm

# Prompt templates
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

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "chat_history", "question"])

def set_no_docs_prompt():
    return PromptTemplate(template=NO_DOCS_PROMPT, input_variables=["chat_history", "question"])

# Load database and LLM
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
llm = load_llm(HUGGINGFACE_REPO_ID)

# Create custom prompts
qa_prompt = set_custom_prompt()
no_docs_prompt = set_no_docs_prompt()

# In-memory cache for active sessions
active_sessions = {}

def get_session_path(user_id):
    """Generate a file path for the user's session file."""
    return SESSIONS_DIR / f"{user_id}.json"

def save_session(user_id, memory):
    """Persist session memory to disk."""
    session_path = get_session_path(user_id)
    
    # Convert memory messages to serializable format
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    serialized_messages = messages_to_dict(chat_history)
    
    # Add metadata for session management
    session_data = {
        "last_active": datetime.now().isoformat(),
        "messages": serialized_messages
    }
    
    with open(session_path, 'w') as f:
        json.dump(session_data, f)

def load_session(user_id):
    """Load session memory from disk or create a new one."""
    session_path = get_session_path(user_id)
    
    # Create a new memory instance
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    # Try to load existing session
    if session_path.exists():
        try:
            with open(session_path, 'r') as f:
                session_data = json.load(f)
            
            # Update last active timestamp
            session_data["last_active"] = datetime.now().isoformat()
            
            # Deserialize messages and add to memory
            messages = messages_from_dict(session_data["messages"])
            for message in messages:
                memory.chat_memory.add_message(message)
                
            # Update the session file with new timestamp
            with open(session_path, 'w') as f:
                json.dump(session_data, f)
                
            print(f"Loaded existing session for user: {user_id}")
        except Exception as e:
            print(f"Error loading session for user {user_id}: {e}")
            # Continue with new memory if loading fails
    else:
        print(f"Creating new session for user: {user_id}")
    
    return memory

def get_session_memory(user_id):
    """Get memory for the user, either from cache or disk."""
    # Check if session is in memory cache first
    if user_id in active_sessions:
        print(f"Using cached session for user: {user_id}")
        return active_sessions[user_id]
    
    # Load from disk or create new
    memory = load_session(user_id)
    
    # Store in cache for future requests
    active_sessions[user_id] = memory
    return memory

def clean_llm_response(text):
    """Clean the LLM response but preserve content."""
    prefixes = [r'^Assistant:', r'^YOUR ANSWER:']
    for prefix in prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE).strip()
    
    return text

def cleanup_expired_sessions():
    """Remove session files that haven't been used in SESSION_EXPIRY_DAYS."""
    now = datetime.now()
    count = 0
    
    for session_file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            last_active = datetime.fromisoformat(session_data["last_active"])
            days_since_active = (now - last_active).days
            
            if days_since_active > SESSION_EXPIRY_DAYS:
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
    
    # Get Firebase ID token from request
    id_token = data.get('idToken')
    
    # Fallback session_id (for testing or if Firebase auth fails)
    fallback_session_id = data.get('session_id', 'anonymous')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    # Verify Firebase token and get user ID
    user_id = None
    if id_token and firebase_enabled:
        user_id = verify_firebase_token(id_token)
        if not user_id:
            return jsonify({"error": "Invalid Firebase ID token"}), 401
    else:
        # Fall back to session_id if Firebase is not enabled or token not provided
        user_id = fallback_session_id
    
    # Get or create user's memory
    memory = get_session_memory(user_id)
    
    # Create QA chain with the user-specific memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={'k': 3, 'score_threshold': 0.5}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        get_chat_history=lambda h: h,
        verbose=True
    )

    # Get response from chain
    chain_response = qa_chain.invoke({'question': user_query})
    
    # Get source documents
    source_docs = chain_response.get("source_documents", [])
    
    # Extract and clean the current answer
    current_answer = chain_response.get("answer", "I'm not sure how to respond to that.")
    cleaned_answer = clean_llm_response(current_answer)
    
    # If no source documents were found, use the general knowledge prompt
    if not source_docs:
        print(f"No relevant documents found for user {user_id}, using general knowledge prompt")
        chat_history = memory.load_memory_variables({})["chat_history"]
        general_response = llm.invoke(no_docs_prompt.format(
            chat_history=chat_history,
            question=user_query
        ))
        cleaned_answer = clean_llm_response(general_response)
    
    # Save updated session to disk
    save_session(user_id, memory)
    
    print(f"USER: {user_id}")
    print("CLEANED RESULT: ", cleaned_answer)
    print("SOURCE DOCUMENTS COUNT: ", len(source_docs))

    return jsonify({
        "response": cleaned_answer,
        "has_source_documents": len(source_docs) > 0,
        "user_id": user_id  # Return the user ID that was used
    })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    data = request.json
    
    # Get Firebase ID token from request
    id_token = data.get('idToken')
    fallback_session_id = data.get('session_id', 'anonymous')
    
    # Verify Firebase token and get user ID
    user_id = None
    if id_token and firebase_enabled:
        user_id = verify_firebase_token(id_token)
        if not user_id:
            return jsonify({"error": "Invalid Firebase ID token"}), 401
    else:
        user_id = fallback_session_id
    
    # Remove from memory cache
    if user_id in active_sessions:
        del active_sessions[user_id]
    
    # Remove from disk
    session_path = get_session_path(user_id)
    if session_path.exists():
        session_path.unlink()
        return jsonify({"message": f"History for user {user_id} cleared successfully"})
    
    return jsonify({"message": f"No history found for user {user_id}"})

@app.route('/admin/active_users', methods=['GET'])
def list_active_users():
    # This should be properly secured in production
    if not request.headers.get('X-Admin-Key') == os.environ.get('ADMIN_API_KEY', 'admin-secret'):
        return jsonify({"error": "Unauthorized"}), 401
        
    # List all users, both in memory and on disk
    disk_users = [f.stem for f in SESSIONS_DIR.glob("*.json")]
    
    return jsonify({
        "in_memory_users": list(active_sessions.keys()),
        "all_users": disk_users,
        "memory_count": len(active_sessions),
        "total_count": len(disk_users)
    })

@app.route('/admin/cleanup_sessions', methods=['POST'])
def cleanup_sessions():
    """Endpoint to trigger cleanup of expired sessions."""
    if not request.headers.get('X-Admin-Key') == os.environ.get('ADMIN_API_KEY', 'admin-secret'):
        return jsonify({"error": "Unauthorized"}), 401
        
    deleted_count = cleanup_expired_sessions()
    return jsonify({"message": f"Cleaned up {deleted_count} expired sessions"})

# For health checks
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "firebase_enabled": firebase_enabled})

# Schedule periodic cleanup (optional - can also be done with a cron job)
# @app.before_first_request
# def schedule_cleanup():
#     # This will run cleanup once when the app starts
#     cleanup_expired_sessions()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)