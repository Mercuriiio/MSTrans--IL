import torch
import torch.nn as nn
from sparse_attention import SparseAttention

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Ensure the embedding size is divisible by number of heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Linear layers for the queries, keys, and values
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)

        # Output fully connected layer
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Transform the input using the query, key, and value linear layers
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Split the embedding size into the number of heads
        # This allows for multiple attention scores
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Scaled dot-product attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Turn the energy values into probabilities ranging from 0 to 1
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Multiply the attention scores with the values to get the final output
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # Apply the output linear layer
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SparseAttention(4, "strided", 8, 8)  # local_attn_ctx * blocksize = 64

        self.feed_forward = nn.Sequential(
            nn.Linear(64*256, 256),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(query, key, value)  # [B, 64, 256]
        # print(attention.shape)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(attention + query)
        out = self.feed_forward(x.view(x.shape[0], 64*256))
        return out, attention


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Encoder, self).__init__()

        # Set the embedding size and device
        self.embed_size = embed_size
        self.device = device

        # Initialize encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Pass the input through the encoder layers
        for layer in self.layers:
            out, attn = layer(x, x, x, mask)

        return out, attn


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        src_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):
        super(Transformer, self).__init__()

        # Encoder to process the source sequence
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        # Define padding indices for source and target sequences
        self.src_pad_idx = src_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # Create mask for source sequence to ignore padding tokens
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # Create mask for target sequence to avoid attending to future tokens
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src):
        # Create masks
        src_mask = self.make_src_mask(src)

        # Process source sequence with encoder
        enc_src, attn = self.encoder(src, src_mask)

        return enc_src, attn

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.randn(16, 64, 256).to(device)

    src_pad_idx = 0
    src_vocab_size = 256

    model = Transformer(src_vocab_size, src_pad_idx, device=device).to(device)
    out, attn = model(x)
    print(out.shape, attn.shape)