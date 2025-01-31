from flask import Flask, render_template, request, jsonify
from chatbot_inference import predict_response  # Import your predict function

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # This will serve your HTML page

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']  # Get the message from the user
    bot_response = predict_response(user_message)  # Get the bot's response
    return jsonify({'response': bot_response})  # Return the response in JSON format

if __name__ == '__main__':
    app.run(debug=True)
