from flask import Flask, request, jsonify
from retrieve_chatengine_Tiabete import get_chat_engine 
from config import RAGConfig
import json
import os

app = Flask(__name__)
chat_engine = get_chat_engine()

config = RAGConfig

SECRET_API_KEY = config.SECRET_API_KEY

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    api_key = request.headers.get("X-API-KEY")

    # Verifica se a API Key foi fornecida e é válida
    if not api_key or api_key != SECRET_API_KEY:
        return jsonify({"error": "Unauthorized - Invalid API Key"}), 401
    
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response = chat_engine.chat(user_message)
    
    # Converte o objeto em texto
    final_response = response.response if hasattr(response, 'response') else str(response)
    
    return app.response_class(
    response=json.dumps({"response": final_response}, ensure_ascii=False),  # Garante caracteres UTF-8
    status=200,
    mimetype="application/json; charset=utf-8"
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))  # Render fornece a variável PORT
    app.run(host="0.0.0.0", port=port, debug=True)