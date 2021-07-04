from flask import Flask,render_template,request,flash,redirect, url_for
import os
import chatbot
from twilio.twiml.messaging_response import MessagingResponse
from werkzeug.utils import redirect, secure_filename


# Create flask app
app = Flask(__name__)
# for encrypting the session
app.secret_key = "secret key" 

@app.route("/" )
def hello():
    return render_template("index.html")

 #route to accept the chat queries 
@app.route("/bot", methods = ["POST"])
def chat():
    if request.method=="POST":
        response = MessagingResponse()
        msg=request.form.get('Body').lower()
        # if mpty string is received
        if msg == '':
            response.message(chatbot.chatbot_response("Hi"))
            # return chatbot.chatbot_response("Hi")
            
        # if string is not empty
        response.message(chatbot.chatbot_response(msg))
        return str(response)
        # return chatbot.chatbot_response(msg)

if __name__ == "__main__":
    app.run(port=5000,threaded=True)