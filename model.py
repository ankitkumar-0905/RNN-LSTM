import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim
    ):
        super().__init__()

        # Word IDs → Embedding vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Classification layer
        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )


    def forward(self, x):

        # [batch, sequence_length]
        x = self.embedding(x)

        # [batch, sequence_length, embedding_dim]
        output, (hidden, cell) = self.lstm(x)

        # Last hidden state
        x = hidden[-1]

        # Final classification
        x = self.fc(x)

        return x