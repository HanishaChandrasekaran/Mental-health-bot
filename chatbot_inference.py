import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained("./model")

# Define responses for different labels (stress, depression, etc.)
responses = {
    0: [
        "Work stress is common. Maybe some time management strategies could help?",
        "Managing work stress can be tough. How about organizing your tasks?",
        "Sometimes, breaking tasks into smaller chunks can reduce stress."
    ],
    1: [
        "I understand that stress can be overwhelming. Would you like some tips on managing it?",
        "Stress is really challenging, but we can talk about ways to cope with it. What do you think?",
        "Taking breaks and talking it out might help. Want to share more about what's stressing you?"
    ],
    2: [
        "It sounds like you're feeling a bit down. Do you want to talk about it?",
        "Depression can be really tough. Would you like some advice on managing it?",
        "Sometimes it helps to talk it out. What’s been on your mind?"
    ],
    3: [
        "Anxiety can make everything feel uncertain. Would you like to try some relaxation exercises?",
        "I understand anxiety can be overwhelming. Want some tips to calm down?",
        "It’s okay to feel anxious sometimes. How about we focus on deep breathing exercises?"
    ],
    # Add more responses for other labels here
}

# Function to ensure we don't repeat the same response
last_response = None

def get_response(label):
    global last_response
    response_list = responses.get(label, ["I'm here to listen whenever you're ready."])
    
    # Ensure we don't repeat the last response
    response = random.choice(response_list)
    while response == last_response:
        response = random.choice(response_list)
    
    last_response = response
    return response

def predict_response(conversation):
    # Tokenize and predict the conversation
    inputs = tokenizer(conversation, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted label
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    
    # Get a dynamic response based on the predicted label
    response = get_response(predicted_label)
    
    return response

if __name__ == "__main__":
    print("Welcome to the Mental Health Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Take care! Bye!")
            break
        response = predict_response(user_input)
        print(f"Bot: {response}")
