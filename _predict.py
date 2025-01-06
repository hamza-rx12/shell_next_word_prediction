
import sys
import tensorflow as tf

with tf.device('/device:GPU:0'):

    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np
    import pickle
    
    # Load data
    commands = []
    with open('/home/natsu/Desktop/shell_next_word_prediction/lstm/data.txt', 'r') as file:
        for line in file:
            if line:
                commands.append(line.strip())
    
    # Load tokenizer
    with open("/home/natsu/Desktop/shell_next_word_prediction/lstm/lstm_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    # Load model
    model = load_model('/home/natsu/Desktop/shell_next_word_prediction/lstm/lstm_model.keras')
        
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

    # Pad sequences
    max_seq_len = max(len(seq) for seq in input_sequences)

    # Prediction function
    def predict_next_word(model, tokenizer, input_text, max_seq_len):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_seq_len, padding='post')
        prediction = model.predict(sequence)
        predicted_index = np.argmax(prediction)
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                return word, prediction[0][index]
        return None


if __name__ == "__main__":
    print(predict_next_word(model,tokenizer," ".join(sys.argv[1:]), max_seq_len)[0])

