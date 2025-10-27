import uuid
from datetime import datetime

# In-memory chat storage
chats = {}

def create_chat():
    chat_id = str(uuid.uuid4())
    chats[chat_id] = []
    return chat_id

def add_message(chat_id, message, role):
    if chat_id in chats:
        msg_obj = {
            "role": role,  # 'patient' or 'doctor'
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        chats[chat_id].append(msg_obj)

def get_chat(chat_id):
    return chats.get(chat_id)

def get_all_chats():
    return [{"chat_id": cid, "message_count": len(msgs)} for cid, msgs in chats.items()]
