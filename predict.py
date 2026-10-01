import re
import json

import torch

from model import LSTMModel


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# 2. Load Vocabulary
# ==========================================

with open(
    "vocabulary.json",
    "r"
) as file:

    word_to_idx = json.load(file)


print(
    "Vocabulary Size:",
    len(word_to_idx)
)


# ==========================================
# 3. Model Configuration
# ==========================================

MAX_LENGTH = 8

embedding_dim = 32
hidden_dim = 64
output_dim = 2


# ==========================================
# 4. Create Model
# ==========================================

model = LSTMModel(
    vocab_size=len(word_to_idx),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim
)


# ==========================================
# 5. Load Trained Weights
# ==========================================

model.load_state_dict(
    torch.load(
        "lstm_model.pth",
        map_location=device
    )
)


model = model.to(device)

model.eval()


print("\nModel loaded successfully!")


# ==========================================
# 6. Tokenization
# ==========================================

def tokenize(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    return text.split()


# ==========================================
# 7. Encode Text
# ==========================================

def encode(tokens):

    return [
        word_to_idx.get(
            word,
            word_to_idx["<UNK>"]
        )
        for word in tokens
    ]


# ==========================================
# 8. Padding
# ==========================================

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


# ==========================================
# 9. Prediction Function
# ==========================================

def predict_sentiment(text):

    # Tokenization
    tokens = tokenize(text)

    # Convert words → numbers
    encoded = encode(tokens)

    # Padding
    padded = pad_sequence(
        encoded,
        MAX_LENGTH
    )

    # Convert to Tensor
    input_tensor = torch.tensor(
        [padded],
        dtype=torch.long
    ).to(device)


    # Model prediction
    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            output,
            dim=1
        ).item()


    # Labels
    if prediction == 1:

        sentiment = "Positive"

    else:

        sentiment = "Negative"


    confidence = (
        probabilities[0][prediction]
        .item()
        * 100
    )


    return sentiment, confidence


# ==========================================
# 10. Test Sentences
# ==========================================

sentences = [
    "this movie was amazing",
    "i loved this film",
    "this movie was terrible",
    "i hated this movie",
    "the movie was boring"
]


for sentence in sentences:

    sentiment, confidence = (
        predict_sentiment(sentence)
    )


    print("\nText:", sentence)

    print(
        "Prediction:",
        sentiment
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )


# ==========================================
# 11. Custom Input
# ==========================================

while True:

    text = input(
        "\nEnter a sentence "
        "(type 'exit' to quit): "
    )


    if text.lower() == "exit":

        print("Program stopped.")

        break


    sentiment, confidence = (
        predict_sentiment(text)
    )


    print(
        "Prediction:",
        sentiment
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )