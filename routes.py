from flask import render_template, request, jsonify
from app import app
from app.chatbot_logic import generate_response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': "Please enter a message."})

    response = generate_response(user_input)
    return jsonify({'response': response})
