import openai
from os import environ, path
from dotenv import load_dotenv
import requests
import uuid
from flask import Flask, request, jsonify, send_file, render_template

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, ".env"))

openai.api_key = environ.get("OPENAI_API_KEY")

app = Flask(__name__)
# app.config["FLASK_ENV"] = "development"
app.config["SECRET_KEY"] = environ.get("SECRET_KEY")


def transcribe_audio(filename: str) -> str:
    """Transcribe audio to text.

    :param filename: The path to an audio file.
    :returns: The transcribed text of the file.
    :rtype: str

    """
    with open(filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript.text


def generate_reply(conversation: list) -> str:
    """Generate a ChatGPT response.

    :param conversation: A list of previous user and assistant messages.
    :returns: The ChatGPT response.
    :rtype: str

    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a expert in computer science and Java. You will explain everything so that even beginners can understand and go in depth if asked. You will provide code examples if applicable.",
            },
        ]
        + conversation,
    )
    return response["choices"][0]["message"]["content"]


@app.route("/")
def index():
    """Render the index page."""
    return render_template("index.html", title="Home", footer="fixed-bottom")


@app.route("/cmsc131")
def cmsc131():
    return render_template("cmsc131.html", title="CMSC 131")


@app.route("/cmsc132")
def cmsc132():
    return render_template("cmsc132.html", title="CMSC 132")


@app.route("/resources")
def resources():
    return render_template("resources.html", title="Resources", footer="fixed-bottom")


@app.route("/chat")
def chat():
    return render_template("chat.html", title="Chat", footer="fixed-bottom")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribe the given audio to text using Whisper."""
    if "file" not in request.files:
        return "No file found", 400
    file = request.files["file"]
    recording_file = f"{uuid.uuid4()}.wav"
    recording_path = f"uploads/{recording_file}"
    os.makedirs(os.path.dirname(recording_path), exist_ok=True)
    file.save(recording_path)
    transcription = transcribe_audio(recording_path)
    return jsonify({"text": transcription})


@app.route("/ask", methods=["POST"])
def ask():
    """Generate a ChatGPT response from the given conversation"""
    conversation = request.get_json(force=True).get("conversation", "")
    reply = generate_reply(conversation)
    return jsonify({"text": reply})


@app.route("/listen/<filename>")
def listen(filename):
    """Return the audio file located at the given filename."""
    return send_file(f"outputs/{filename}", mimetype="audio/mp3", as_attachment=False)


if __name__ == "__main__":
    app.run()
