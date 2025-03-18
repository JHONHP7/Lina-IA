from flask import Flask, request, jsonify
from retrieve_chatengine_Tiabete import get_chat_engine 

app = Flask(__name__)
chat_engine = get_chat_engine()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response = chat_engine.chat(user_message)
    
    # Aqui, convertemos o objeto em texto
    if hasattr(response, 'response'):  # Checa se o objeto tem um atributo response (padrão no LlamaIndex)
        final_response = response.response
    else:
        final_response = str(response)  # fallback para string

    return jsonify({"response": final_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
