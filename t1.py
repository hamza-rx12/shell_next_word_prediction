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

with open("/home/natsu/Desktop/shell_next_word_prediction/lstm/lstm_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model('/home/natsu/Desktop/shell_next_word_prediction/lstm/lstm_model.keras')

MAX_SEQ_LEN = 20

def predict_next_words(model, tokenizer, input_text, max_seq_len, top_k=5):
    sequence = tokenizer.texts_to_sequences([input_text])
    sequence = pad_sequences(sequence, maxlen=max_seq_len, padding='post')
    prediction = model.predict(sequence)
    probabilities = prediction[0]
    top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
    suggestions = [word for word, index in tokenizer.word_index.items() if index in top_k_indices]
    return suggestions



def prompt_autocomplete():
    app = get_app()
    b = app.current_buffer
    if b.complete_state:
        b.complete_next()
    else:
        b.start_completion(select_first=False)

command = []

if __name__ == "__main__":
    # Capture the current command passed via sys.argv
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <partial_command>")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    command.append(input_text)

    while True:

        words = input_text.split()

        if len(words) > 1:
            # Provide the last two words for better context
            input_text = " ".join(words[-1:])
        else:
            # If only one word, just use the single word
            input_text = words[-1]

        suggestions = predict_next_words(model, tokenizer, " ".join(command), MAX_SEQ_LEN)

        if not suggestions:
            print("No suggestions available.")
            sys.exit(0)

        # Use prompt_toolkit to display suggestions and allow user interaction
        completer = WordCompleter(suggestions, ignore_case=True)
        completed_command = prompt(f"{" ".join(command)}", pre_run=prompt_autocomplete, completer=completer)
        command.append(completed_command)

        print(" ".join(command))

