from flask import Blueprint, request, jsonify
from models.chat_model import create_chat, add_message, get_all_chats, get_chat

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


    add_message(chat_id, message, 'doctor')

    response = "gad to meet you doctor"
    add_message(chat_id, response, 'bot')

    return jsonify({"response": response, "role":"bot"})


@chat_bp.route("/get_messages/<chat_id>", methods=["GET"])
def get_messages(chat_id):
    messages = get_chat(chat_id)
    if messages is None:
        return jsonify({"error": "Chat not found"}), 404
    return jsonify({"chat_id": chat_id, "messages": messages})