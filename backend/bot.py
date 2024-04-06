from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-large"  # Change to a larger model like "gpt2-large" or "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

CODING_CONTEXT = "I am an AI assistant specializing in providing code suggestions."


def generate_response(input_text):
    input_with_context = f"{CODING_CONTEXT} {input_text}"
    input_ids = tokenizer.encode(input_with_context, return_tensors="pt")

    if input_ids is None:
        response = "I'm sorry, I couldn't understand your input. Please rephrase your question or provide more context."
        return response

    if tokenizer.pad_token_id is None:
        attention_mask = None
    else:
        attention_mask = input_ids.ne(tokenizer.pad_token_id).float()

    # Generate response
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=500, num_return_sequences=1,
                            do_sample=True, top_k=50, top_p=0.95, num_beams=5, pad_token_id=tokenizer.eos_token_id)

    # Decode response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


@app.route('/chatbot', methods=['POST'])
def chatbot():
    # chat
    data = request.get_json()
    user_input = data['input']

    # Generate full response
    full_response = generate_response(user_input)

    return jsonify({'response': full_response})


if __name__ == '__main__':
    app.run(debug=True)
