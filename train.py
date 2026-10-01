import re
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from model import LSTMModel


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# 2. Load CSV Dataset
# ==========================================

df = pd.read_csv("data/reviews.csv")

print("\nDataset:")
print(df)

print("\nDataset Shape:")
print(df.shape)


# ==========================================
# 3. Tokenization
# ==========================================

def tokenize(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    return text.split()


df["tokens"] = df["text"].apply(tokenize)


# ==========================================
# 4. Build Vocabulary
# ==========================================

counter = Counter()

for tokens in df["tokens"]:
    counter.update(tokens)


word_to_idx = {
    "<PAD>": 0,
    "<UNK>": 1
}


for word in counter:
    word_to_idx[word] = len(word_to_idx)


print("\nVocabulary:")
print(word_to_idx)

print(
    "\nVocabulary Size:",
    len(word_to_idx)
)


# ==========================================
# 5. Convert Words → Numbers
# ==========================================

def encode(tokens):

    return [
        word_to_idx.get(
            word,
            word_to_idx["<UNK>"]
        )
        for word in tokens
    ]


df["encoded"] = df["tokens"].apply(encode)


# ==========================================
# 6. Padding
# ==========================================

MAX_LENGTH = 8


def pad_sequence(
    sequence,
    max_length
):

    if len(sequence) < max_length:

        sequence = (
            sequence
            + [0] * (
                max_length
                - len(sequence)
            )
        )

    else:

        sequence = sequence[
            :max_length
        ]

    return sequence


df["padded"] = df["encoded"].apply(
    lambda x: pad_sequence(
        x,
        MAX_LENGTH
    )
)


# ==========================================
# 7. Create X and y
# ==========================================

X = torch.tensor(
    df["padded"].tolist(),
    dtype=torch.long
)

y = torch.tensor(
    df["label"].tolist(),
    dtype=torch.long
)


print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)


# ==========================================
# 8. Train / Validation / Test Split
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nTrain Samples:", len(X_train))
print("Validation Samples:", len(X_val))
print("Test Samples:", len(X_test))


# ==========================================
# 9. Dataset Class
# ==========================================

class ReviewDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):
        self.X = X
        self.y = y


    def __len__(self):

        return len(self.X)


    def __getitem__(self, index):

        return (
            self.X[index],
            self.y[index]
        )


# ==========================================
# 10. Create Dataset
# ==========================================

train_dataset = ReviewDataset(
    X_train,
    y_train
)

val_dataset = ReviewDataset(
    X_val,
    y_val
)

test_dataset = ReviewDataset(
    X_test,
    y_test
)


# ==========================================
# 11. DataLoader
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)


# ==========================================
# 12. Create LSTM Model
# ==========================================

model = LSTMModel(
    vocab_size=len(word_to_idx),
    embedding_dim=32,
    hidden_dim=64,
    output_dim=2
)


model = model.to(device)


# ==========================================
# 13. Loss Function
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# 14. Optimizer
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 15. Training Settings
# ==========================================

epochs = 30

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


# ==========================================
# 16. Training Loop
# ==========================================

for epoch in range(epochs):

    # ------------------------------
    # Training
    # ------------------------------

    model.train()

    total_train_loss = 0

    correct_train = 0
    total_train = 0


    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)


        # Forward pass
        output = model(batch_X)


        # Calculate loss
        loss = criterion(
            output,
            batch_y
        )


        # Clear gradients
        optimizer.zero_grad()


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Statistics
        total_train_loss += loss.item()


        predictions = torch.argmax(
            output,
            dim=1
        )


        correct_train += (
            predictions == batch_y
        ).sum().item()


        total_train += batch_y.size(0)


    train_loss = (
        total_train_loss
        / len(train_loader)
    )


    train_accuracy = (
        correct_train
        / total_train
    )


    # ------------------------------
    # Validation
    # ------------------------------

    model.eval()

    total_val_loss = 0

    correct_val = 0
    total_val = 0


    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)


            output = model(batch_X)


            loss = criterion(
                output,
                batch_y
            )


            total_val_loss += (
                loss.item()
            )


            predictions = torch.argmax(
                output,
                dim=1
            )


            correct_val += (
                predictions == batch_y
            ).sum().item()


            total_val += batch_y.size(0)


    val_loss = (
        total_val_loss
        / len(val_loader)
    )


    val_accuracy = (
        correct_val
        / total_val
    )


    # Save metrics
    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy * 100:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )


# ==========================================
# 17. Test Model
# ==========================================

model.eval()

correct_test = 0
total_test = 0


with torch.no_grad():

    for batch_X, batch_y in test_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)


        output = model(batch_X)


        predictions = torch.argmax(
            output,
            dim=1
        )


        correct_test += (
            predictions == batch_y
        ).sum().item()


        total_test += batch_y.size(0)


test_accuracy = (
    correct_test
    / total_test
)


print(
    "\nTest Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


# ==========================================
# 18. Save Model
# ==========================================

torch.save(
    model.state_dict(),
    "lstm_model.pth"
)


print("\nModel saved as: lstm_model.pth")


# ==========================================
# 19. Save Vocabulary
# ==========================================

import json

with open(
    "vocabulary.json",
    "w"
) as file:

    json.dump(
        word_to_idx,
        file
    )


print(
    "Vocabulary saved as: vocabulary.json"
)


# ==========================================
# 20. Loss Graph
# ==========================================

plt.figure()

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.show()


# ==========================================
# 21. Accuracy Graph
# ==========================================

plt.figure()

plt.plot(
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.show()