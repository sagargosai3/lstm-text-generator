import torch
import torch.nn as nn
import numpy as np
import string
from collections import Counter

# 1. Load & preprocess text
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()

    print(f"Total words: {len(words)}")
    return words


# 2. Build vocabulary
def build_vocab(words):
    word_counts = Counter(words)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    vocab_size = len(word_to_idx) + 1  # +1 for padding

    print(f"Vocab size: {vocab_size}")
    return word_to_idx, idx_to_word, vocab_size


# 3. Create sequences
def create_sequences(words, word_to_idx, seq_len=5):
    sequences = []

    for i in range(seq_len, len(words)):
        seq = words[i - seq_len:i + 1]
        sequences.append([word_to_idx[w] for w in seq])

    sequences = np.array(sequences)

    X = sequences[:, :-1]
    y = sequences[:, -1]

    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# 4. Define LSTM Model
class TextGenerator(nn.Module):
    def __init__(self, vocab_size):
        super(TextGenerator, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(input_size=100, hidden_size=150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # last timestep
        output = self.fc(output)
        return output


# 5. Training function
def train_model(model, X, y, epochs=5, batch_size=128):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(X) // batch_size)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")


# 6. Text generation
def generate_text(model, seed_text, word_to_idx, idx_to_word, seq_len=5, next_words=15):
    model.eval()
    words = seed_text.lower().split()

    for _ in range(next_words):
        seq = [word_to_idx.get(w, 0) for w in words[-seq_len:]]
        seq = [0] * (seq_len - len(seq)) + seq

        seq = torch.tensor([seq], dtype=torch.long)

        with torch.no_grad():
            output = model(seq)
            predicted_idx = torch.argmax(output).item()

        next_word = idx_to_word.get(predicted_idx, "")
        words.append(next_word)

    return " ".join(words)


# 7. Main execution
if __name__ == "__main__":
    FILE_PATH = "shakespeare.txt"

    words = load_data(FILE_PATH)
    word_to_idx, idx_to_word, vocab_size = build_vocab(words)

    X, y = create_sequences(words, word_to_idx)

    print("Dataset shape:", X.shape, y.shape)

    model = TextGenerator(vocab_size)

    train_model(model, X, y)

    # Generate sample text
    result = generate_text(
        model,
        seed_text="to be or not to",
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        next_words=20
    )

    print("\nGenerated Text:\n", result)