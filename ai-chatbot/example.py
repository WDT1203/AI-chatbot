# import os
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.chains import ConversationalRetrievalChain

# app = Flask(__name__)
# CORS(app)

# # setup LLM (mistral)
# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id = huggingface_repo_id,
#         temperature = 0.6,
#         huggingfacehub_api_token=HF_TOKEN,
#         max_new_tokens=200,
#         top_k=50,  # Pass top_k directly
#         top_p=0.9,
#         model_kwargs = {
#             "max_length":512,        }
#     )
#     return llm

# # CUSTOM_PROMPT_TEMPLATE to allow for conversational responses
# CUSTOM_PROMPT_TEMPLATE = """
# You are a friendly and knowledgeable assistant. Your goal is to provide helpful, engaging, and conversational responses.

# For questions:
# - Use ONLY the factual information provided in the context below.
# - If the context doesn't contain relevant information, politely inform the user and suggest alternatives.
# - Avoid making up information.

# For general chat:
# - Respond naturally without forcing a structured Q&A format.
# - Be engaging, friendly, and adaptable.

# Context: {context}
# User: {question}
# Assistant:
# """


# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
#     return prompt

# # load database
# DB_FAISS_PATH = "vectorstore/db_faiss" 
# embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Define memory
# memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# # Create a conversational chain with memory
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     retriever=db.as_retriever(search_kwargs={'k': 5, 'score_threshold': 0.5}),
#     memory=memory,
#     return_source_documents=True,
#     output_key="answer"
# )
# # # create QA chain
# # qa_chain = RetrievalQA.from_chain_type(
# #     llm = load_llm(HUGGINGFACE_REPO_ID),
# #     chain_type = "stuff",
# #     retriever = db.as_retriever(search_kwargs = {'k':5, 'score_threshold': 0.5}),
# #     return_source_documents = True,
# #     chain_type_kwargs = {'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# # )

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json
#     user_query = data.get('query')
#     if not user_query:
#         return jsonify({"error": "No query provided"}), 400

#     # Load chat history from memory
#     chat_history = memory.load_memory_variables({}).get('chat_history', [])

#     # Pass both the question and the chat history
#     response = qa_chain.invoke({'question': user_query, 'chat_history': chat_history})

#     print("RESULT: ", response["answer"])
#     print("SOURCE DOCUMENTS: ", response["source_documents"])
#     print("Number of documents in FAISS:", db.index.ntotal)

#     assistant_response = response.get("answer", "I'm not sure how to respond to that.")

#     return jsonify({
#         "response": assistant_response,
#         "source_documents": str(response.get("source_documents"))
#     })


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)