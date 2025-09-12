from flask import Flask,jsonify
from controllers.chatController import upload_bp


from agents.agentsCenter import GlobalAgentState

from agents.chatAgent.buildAgent import builder

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Register the Blueprint
app.register_blueprint(upload_bp)

@app.route("/health", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"}), 200


def AgentInitializer():

    AgentsCenter = GloablAgentState()
    AgentsCenter.chatAgent = builder()

if __name__ == "__main__":

    # AgentInitializer()
    app.run(debug=True, host="0.0.0.0", port=5000)

