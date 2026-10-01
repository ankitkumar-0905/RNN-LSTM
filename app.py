import re
import json
import torch
import streamlit as st

from model import LSTMModel


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="LSTM Sentiment Analyzer",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------
# Model Configuration
# -----------------------------
MAX_LENGTH = 8
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
OUTPUT_DIM = 2

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Load Vocabulary
# -----------------------------
@st.cache_resource
def load_vocabulary():

    with open("vocabulary.json", "r") as file:
        word_to_idx = json.load(file)

    return word_to_idx


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model(vocab_size):

    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM
    )

    model.load_state_dict(
        torch.load(
            "lstm_model.pth",
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model


# -----------------------------
# Tokenization
# -----------------------------
def tokenize(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    return text.split()


# -----------------------------
# Encoding
# -----------------------------
def encode(tokens, word_to_idx):

    return [
        word_to_idx.get(
            word,
            word_to_idx["<UNK>"]
        )
        for word in tokens
    ]


# -----------------------------
# Padding
# -----------------------------
def pad_sequence(sequence):

    if len(sequence) < MAX_LENGTH:

        sequence = sequence + [
            0
        ] * (
            MAX_LENGTH - len(sequence)
        )

    else:

        sequence = sequence[
            :MAX_LENGTH
        ]

    return sequence


# -----------------------------
# Prediction
# -----------------------------
def predict_sentiment(
    text,
    model,
    word_to_idx
):

    tokens = tokenize(text)

    encoded = encode(
        tokens,
        word_to_idx
    )

    padded = pad_sequence(encoded)

    input_tensor = torch.tensor(
        [padded],
        dtype=torch.long
    ).to(device)

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

    if prediction == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    confidence = (
        probabilities[0][prediction].item()
        * 100
    )

    return sentiment, confidence


# -----------------------------
# Load Everything
# -----------------------------
try:

    word_to_idx = load_vocabulary()

    model = load_model(
        len(word_to_idx)
    )

except Exception as e:

    st.error(
        "Model files load nahi ho pa rahe hain."
    )

    st.code(str(e))

    st.stop()


# -----------------------------
# UI
# -----------------------------
st.title("🤖 LSTM Sentiment Analyzer")

st.write(
    "Enter a movie review and the LSTM model "
    "will predict its sentiment."
)


st.info(
    f"Model Device: {device}"
)


# -----------------------------
# Input
# -----------------------------
text = st.text_area(
    "Enter your review:",
    placeholder="Example: this movie was amazing",
    height=120
)


# -----------------------------
# Predict Button
# -----------------------------
if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if not text.strip():

        st.warning(
            "Please enter a review first."
        )

    else:

        sentiment, confidence = predict_sentiment(
            text,
            model,
            word_to_idx
        )

        st.subheader(
            "Prediction"
        )

        if sentiment == "Positive":

            st.success(
                f"😊 {sentiment}"
            )

        else:

            st.error(
                f"😞 {sentiment}"
            )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            confidence / 100
        )


# -----------------------------
# Example Reviews
# -----------------------------
st.divider()

st.subheader(
    "Try Example Reviews"
)

examples = [
    "this movie was amazing",
    "i really loved this film",
    "this movie was terrible",
    "i hated this movie",
    "the movie was boring"
]

for example in examples:

    if st.button(
        example,
        key=example,
        use_container_width=True
    ):

        sentiment, confidence = predict_sentiment(
            example,
            model,
            word_to_idx
        )

        st.write(
            f"**Prediction:** {sentiment}"
        )

        st.write(
            f"**Confidence:** {confidence:.2f}%"
        )