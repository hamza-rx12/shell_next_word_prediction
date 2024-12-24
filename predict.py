import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

commands = []

with open('lstm/data.txt', 'r') as file:
    for line in file:
      if line:
        commands.append(line.strip())

# Load tokenizer
import pickle
with open("/home/natsu/Desktop/shell_next_word_prediction/lstm/lstm_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Convert commands to sequences
sequences = tokenizer.texts_to_sequences(commands)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Create input-output pairs
input_sequences = []
target_sequences = []

for seq in sequences:
    for i in range(1, len(seq)):
        input_sequences.append(seq[:i])
        target_sequences.append(seq[i])

model = load_model('/home/natsu/Desktop/shell_next_word_prediction/lstm/lstm_model.keras')

# Pad sequences
max_seq_len = max(len(seq) for seq in input_sequences)


# def predict_next_word(model, tokenizer, input_text, max_seq_len):
#     sequence = tokenizer.texts_to_sequences([input_text])[0]
#     sequence = pad_sequences([sequence], maxlen=max_seq_len, padding='post')
#     prediction = model.predict(sequence)
#     # predicted_index = np.argmax(prediction)
#     # for word, index in tokenizer.word_index.items():
#     #     if index == predicted_index:
#     #         return word, prediction[0][index]
#     # return None
#     probabilities = prediction[0]
#     top_k_indices = np.argsort(probabilities)[5:][::-1]
#     suggestions = [word for word, index in tokenizer.word_index.items() if index in top_k_indices]
#     return suggestions
# input_text = 'sudo apt install'

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


# print(f"Next word prediction: {next_word}")



if __name__ == "__main__":

    # suggestions = []
    suggestions = predict_next_words(model,tokenizer," ".join(sys.argv[1:]), max_seq_len)[0]
    completer = WordCompleter(suggestions, ignore_case=True)
    command = prompt(completer=completer)
    print(command)
    # if next_word: 
    # print("apt")
    #     print(next_word)