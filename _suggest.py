import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.application.current import get_app
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
import numpy as np
import os
import pickle

os.system('stty sane')
# Load tokenizer
with open("/home/natsu/Desktop/shell_next_word_prediction/lstm/lstm_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the pre-trained model
model = load_model('/home/natsu/Desktop/shell_next_word_prediction/lstm/lstm_model.keras')

# Set max sequence length
MAX_SEQ_LEN = 20 # Adjust based on your model's training configuration

def predict_next_words(model, tokenizer, input_text, max_seq_len, top_k=5):
    """Predict the top-k next words based on the input text."""

    # Tokenize and pad the input sequence
    sequence = tokenizer.texts_to_sequences([input_text])
    sequence = pad_sequences(sequence, maxlen=max_seq_len, padding='post')

    # Predict the next word based on the current sequence
    prediction = model.predict(sequence)
    probabilities = prediction[0]

    # Get top-k predictions
    top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
    
    # Retrieve the top-k predicted words
    suggestions = [word for word, index in tokenizer.word_index.items() if index in top_k_indices]
    return suggestions



def prompt_autocomplete():
    app = get_app()
    b = app.current_buffer
    if b.complete_state:
        b.complete_next()
    else:
        b.start_completion(select_first=False)

# bindings = KeyBindings()

# @bindings.add('ñ')
# def _(event):
#     event.app.exit()

# # Create the application
# application = Application(
#     layout=None,  # Set layout as needed, like a simple `prompt_layout`
#     key_bindings=bindings,
#     full_screen=True
# )


if __name__ == "__main__":
    # Capture the current command passed via sys.argv
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <partial_command>")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    
    # Split the input into words to provide better context
    words = input_text.split()

    # Check if the input has at least one word for prediction
    if len(words) > 1:
        # Provide the last two words for better context
        input_text = " ".join(words[-1:])
    else:
        # If only one word, just use the single word
        input_text = words[-1]
    
    # Get predictions
    suggestions = predict_next_words(model, tokenizer, input_text, MAX_SEQ_LEN)

    if not suggestions:
        print("No suggestions available.")
        sys.exit(0)

    # Use prompt_toolkit to display suggestions and allow user interaction
    completer = WordCompleter(suggestions, ignore_case=True)
    # prompt(pre_run=prompt_autocomplete)
    completed_command = prompt(f"{input_text} ", pre_run=prompt_autocomplete, completer=completer)
    # prompt(pre_run=prompt_autocomplete)

    # Print the final completed command
    print(completed_command)
