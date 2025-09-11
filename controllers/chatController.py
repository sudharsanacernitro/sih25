import os
from flask import Blueprint, request, jsonify, current_app

from services.chatService import handleChatService

# 👇 Give the blueprint a base name and URL prefix
upload_bp = Blueprint("upload", __name__, url_prefix="/chat")

@upload_bp.route("/upload", methods=["POST"])
def upload():
    message = request.form.get("message", "")
    image = request.files.get("image")
    
    result = handleChatService( message , image)

    return jsonify(result)

