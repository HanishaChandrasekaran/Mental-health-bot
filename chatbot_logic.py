from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer from the saved directory
MODEL_PATH = r'D:\health_bot_results'  # Make sure this path exists and contains model files

try:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    print("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Function to generate a response with updated parameters
def generate_response(user_input):
    try:
        # Encode the user input to tensor
        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

        # Generate response using the trained model
        outputs = model.generate(
            inputs,
            max_length=200,  # Increase length for more detailed responses
            num_return_sequences=1,
            no_repeat_ngram_size=3,  # Reduce repetitive text
            top_k=50,  # Can adjust this to control diversity
            temperature=0.9,  # Higher temperature for more creative responses
            pad_token_id=tokenizer.eos_token_id  # Ensure padding token is handled
        )

        # Decode the generated response and return it
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "Sorry, I couldn't generate a response. Try again later."
