from flask import Flask,jsonify
from controllers.chatController import upload_bp

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Register the Blueprint
app.register_blueprint(upload_bp)

@app.route("/health", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

