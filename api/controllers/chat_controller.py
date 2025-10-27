from flask import Blueprint, request, jsonify
from models.chat_model import create_chat, add_message, get_all_chats, get_chat
from scripts.sympto import get_random_true_case, ask_openai, evaluate_conversation 

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/start_chat", methods=["POST"])
def start_chat():
    chat_id = create_chat()
    return jsonify({"chat_id": chat_id, "message": "New chat started!"})

@chat_bp.route("/chats", methods=["GET"])
def list_chats():
    return jsonify(get_all_chats())

@chat_bp.route("/send_message", methods=["POST"])
def send_message():
    data = request.get_json()
    chat_id = data.get("chat_id")
    message = data.get("message")

    if not chat_id or not message:
        return jsonify({"error": "Missing chat_id or message"}), 400

    add_message(chat_id, message, 'doctor')

    response = ask_openai(chat_id, message)

    return jsonify({"response": response, "role":"bot"})


@chat_bp.route("/get_messages/<chat_id>", methods=["GET"])
def get_messages(chat_id):
    messages = get_chat(chat_id)
    if messages is None:
        return jsonify({"error": "Chat not found"}), 404
    return jsonify({"chat_id": chat_id, "messages": messages})

@chat_bp.route("/generate_case/<chat_id>", methods=["GET"])
def generate_case(chat_id):
    try:
        case = get_random_true_case(chat_id)
        return jsonify({"chat_id": chat_id, "case": case})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@chat_bp.route("/evaluate/<chat_id>", methods=["GET"])
def evaluate_chat(chat_id):
    try:
        report = evaluate_conversation(chat_id)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500  