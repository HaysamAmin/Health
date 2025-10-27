# app.py
from flask import Flask
from api.controllers.chat_controller import chat_bp

app = Flask(__name__)

# Register the blueprint under the /chat prefix
app.register_blueprint(chat_bp, url_prefix="/chat")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
