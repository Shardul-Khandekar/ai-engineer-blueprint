from flask import Flask
from twilio.twiml.voice_response import VoiceResponse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/voice", methods=['GET', 'POST'])
def voice():
    """
    Respond to incoming phone calls with a 'Hello world' message
    """

    print(f"Incoming call received")

    # Initialize the TwiML response
    response = VoiceResponse()

    response.say("Hello from a digital copy of Shardul", voice='alice')

    return str(response)

if __name__ == "__main__":
    app.run(port=8000,debug=True)