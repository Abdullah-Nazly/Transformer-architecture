# transformer.py

import random
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from utils import *


class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


class ReLU(nn.Module):
    """
    ReLU activation function
    ReLU(x) = max(0, x) - returns x if x > 0, otherwise returns 0.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.maximum(x, torch.zeros_like(x))


class LayerNorm(nn.Module):
    """
    Layer Normalization implemented from scratch.
    Normalizes inputs across the feature dimension (last dimension).

    Formula:
    - Mean: μ = (1/d) * Σ(x_i)
    - Variance: σ² = (1/d) * Σ(x_i - μ)²
    - Normalized: x_norm = (x - μ) / sqrt(σ² + ε)
    - Output: y = γ * x_norm + β

    Where:
    - d is the feature dimension
    - ε (eps) is a small constant to prevent division by zero
    - γ (gamma) is a learnable scale parameter
    - β (beta) is a learnable shift parameter
    """

    def __init__(self, d_model, eps=1e-5):
        """
        :param d_model: The feature dimension to normalize over
        :param eps: Small constant to prevent division by zero (default: 1e-5)
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters: scale (gamma) and shift (beta)
        # Initialize gamma to 1 and beta to 0 (standard initialization)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        """
        :param x: Input tensor of shape (..., d_model) where ... can be any dimensions
        :return: Normalized tensor of the same shape
        """
        # Compute mean
        mean = x.mean(dim=-1, keepdim=True)

        # Compute variance
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)

        # Apply learnable scale and shift
        output = self.gamma * x_norm + self.beta

        return output


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_positions,
        d_model,
        d_internal,
        num_classes,
        num_layers,
        use_positional_encoding=True,
        use_causal_mask=False,
        batched_pe=False,
        num_heads=1,
    ):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        :param use_positional_encoding: whether to add positional encodings (default=True)
        :param use_causal_mask: whether to use causal masking for autoregressive language modeling (default=False)
        :param batched_pe: whether to use batched positional encoding (default=False)
        :param num_heads: number of attention heads (default=1 for single-head, >1 for multi-head)
        """
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.use_positional_encoding = use_positional_encoding

        self.positional_encoding = PositionalEncoding(
            d_model, num_positions, batched=batched_pe
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model,
                    d_internal,
                    use_causal_mask=use_causal_mask,
                    num_heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """

        if indices.dim() == 1:
            indices = indices.unsqueeze(0)  # Adds batch dimension if missing

        # --- 1️ Embedding + Positional encoding --- #
        x = self.token_embedding(indices)

        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        attention_maps = []

        # --- 2️ Transformer layers --- #
        for layer in self.layers:
            x, attn = layer(x)
            if attn.dim() == 3 and attn.shape[0] == 1:
                attn = attn.squeeze(0)
            attention_maps.append(attn)

        # --- 3️ Output Projection --- #
        logits = self.output_layer(x)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # --- 4️ Flatten for compatibility --- #
        if log_probs.shape[0] == 1:
            log_probs = log_probs.squeeze(0)

        return log_probs, attention_maps


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, use_causal_mask=True, num_heads=1):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        :param use_causal_mask: If True, applies causal masking for autoregressive language modeling
        :param num_heads: Number of attention heads (1 for single-head, >1 for multi-head)
        """
        super().__init__()

        self.num_heads = num_heads
        self.use_multihead = num_heads > 1

        if self.use_multihead:
            assert (
                d_internal % num_heads == 0
            ), f"d_internal ({d_internal}) must be divisible by num_heads ({num_heads})"
            self.d_head = d_internal // num_heads

        # 1. Q, K, V projections
        self.w_queries = nn.Linear(d_model, d_internal, bias=False)
        self.w_keys = nn.Linear(d_model, d_internal, bias=False)
        self.w_values = nn.Linear(d_model, d_internal, bias=False)

        # 2. Output projection to match d_model size
        self.w_output = nn.Linear(d_internal, d_model, bias=False)

        # 3. Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_internal),  # Linear Projection
            ReLU(),  # Non Linear
            nn.Linear(d_internal, d_model),  # Linear Projection
        )

        self.softmax = nn.Softmax(
            dim=-1
        )  # Softmax over the last dimension. dim=-1 for compatibility with multi-head
        self.attention_dropout = nn.Dropout(
            0.1
        )  # Dropout for attention weights to prevent overfitting

        # Layer Normalization implemented from scratch
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

        self.use_causal_mask = use_causal_mask

    def forward(self, input_vecs):
        batch_size, seq_len, _ = input_vecs.shape

        # ------Self-Attention------ #

        Q = self.w_queries(input_vecs)
        K = self.w_keys(input_vecs)
        V = self.w_values(input_vecs)

        if self.use_multihead:
            # Multi-head attention
            # Reshape to split into multiple heads
            Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

            # Compute attention scores for each head
            attention_score = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(
                self.d_head
            )

            # Apply causal mask if needed
            if self.use_causal_mask:
                causal_mask = torch.triu(
                    torch.ones(
                        seq_len, seq_len, device=input_vecs.device, dtype=torch.bool
                    ),
                    diagonal=1,
                )
                attention_score = attention_score.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            attention_weights = self.softmax(attention_score)
            attention_weights = self.attention_dropout(
                attention_weights
            )  # Apply dropout to prevent overfitting
            attention_output = torch.matmul(attention_weights, V)

            # Concatenate all heads back together
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.view(batch_size, seq_len, -1)

            # Average attention weights across heads for return (for compatibility)
            attention_weights = attention_weights.mean(dim=1)
        else:
            # Single-head attention (original implementation)
            attention_score = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(K.size(-1))

            # Apply causal mask if needed (for autoregressive language modeling)
            if self.use_causal_mask:
                causal_mask = torch.triu(
                    torch.ones(
                        seq_len, seq_len, device=input_vecs.device, dtype=torch.bool
                    ),
                    diagonal=1,
                )
                attention_score = attention_score.masked_fill(
                    causal_mask.unsqueeze(0), float("-inf")
                )

            attention_weights = self.softmax(attention_score)
            attention_weights = self.attention_dropout(
                attention_weights
            )  # Apply dropout to prevent overfitting
            attention_output = torch.matmul(attention_weights, V)

        # project attention_output back to d_model for residual connection
        attention_output = self.w_output(attention_output)

        # ------ Residual Connection + Layer Normalization ------ #
        x = self.layer_norm1(input_vecs + attention_output)

        # ------ Feed Forward Network ------ #
        feed_forward_output = self.feed_forward(x)

        # ------ Residual Connection + Layer Normalization ------ #
        output = self.layer_norm2(x + feed_forward_output)

        return output, attention_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        # Clip position indices to valid range (handle sequences longer than num_positions)
        max_pos = self.emb.num_embeddings - 1
        position_indices = torch.clamp(
            torch.arange(input_size, dtype=torch.long), 0, max_pos
        )

        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(position_indices).unsqueeze(0).to(x.device)
            return x + emb_unsq
        else:
            return x + self.emb(position_indices.to(x.device))


# ========================= #
# TRAIN CLASSIFIER FUNCTION #
# ========================= #


def train_classifier(args, train, dev):
    """
    Trains a Transformer classifier for the letter-counting task.
    """

    # --- Enable batching here --- #
    use_batched_training = False
    batch_size = 16
    # ----------------------------- #

    # --- Hyperparameters --- #
    vocab_size = getattr(args, "vocab_size", 27)
    num_positions = getattr(args, "num_positions", 20)
    d_model = getattr(args, "d_model", 128)
    d_internal = getattr(args, "d_internal", 256)
    num_classes = getattr(args, "num_classes", 3)
    num_layers = getattr(args, "num_layers", 1)
    lr = getattr(args, "lr", 1e-4)
    num_epochs = getattr(args, "num_epochs", 10)
    seed = getattr(args, "seed", 1234)
    use_cuda = getattr(args, "use_cuda", False)

    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(torch.cuda.is_available())  # Should return True

    # --- Model --- #
    model = Transformer(
        vocab_size=vocab_size,
        num_positions=num_positions,
        d_model=d_model,
        d_internal=d_internal,
        num_classes=num_classes,
        num_layers=num_layers,
        use_positional_encoding=True,
    ).to(device)

    # Make positional encoding match training mode
    model.positional_encoding.batched = use_batched_training

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fcn = nn.NLLLoss()

    ex_idxs = list(range(len(train)))

    print(f"\nTraining mode: batched={use_batched_training}, batch_size={batch_size}\n")

    for epoch in range(num_epochs):
        model.train()
        random.shuffle(ex_idxs)
        total_loss = 0.0
        t0 = time.time()

        if not use_batched_training:
            # --- Single-example mode --- #
            for ex_idx in ex_idxs:
                ex = train[ex_idx]
                input = ex.input_tensor.to(device)
                target = ex.output_tensor.to(device)

                log_probs, _ = model(input)
                loss = loss_fcn(log_probs, target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )  # Gradient clipping for stability
                optimizer.step()

                total_loss += loss.item()

        else:
            # --- Batched mode --- #
            def pad_to_len(t: torch.LongTensor, L: int) -> torch.LongTensor:
                if len(t) < L:
                    pad_len = L - len(t)
                    return torch.cat([t, torch.zeros(pad_len, dtype=torch.long)])
                else:
                    return t[:L]

            for i in range(0, len(train), batch_size):
                batch = train[i : i + batch_size]

                max_len = num_positions
                inputs = torch.stack(
                    [pad_to_len(ex.input_tensor, max_len) for ex in batch]
                ).to(device)
                targets = torch.stack(
                    [pad_to_len(ex.output_tensor, max_len) for ex in batch]
                ).to(device)

                log_probs, _ = model(inputs)
                loss = loss_fcn(log_probs.view(-1, num_classes), targets.view(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )  # Gradient clipping for stability
                optimizer.step()

                total_loss += loss.item()

        t1 = time.time()
        avg_loss = total_loss / max(1, len(ex_idxs))
        print(
            f"Epoch {epoch+1}/{num_epochs}  Avg Loss = {avg_loss:.4f}  Time = {t1-t0:.2f}s"
        )

        # --- DEV LOSS --- #
        model.eval()
        with torch.no_grad():
            dev_loss = 0.0
            for ex in dev:
                x = ex.input_tensor.to(device)
                y = ex.output_tensor.to(device)

                log_probs, _ = model(x.unsqueeze(0))
                if log_probs.dim() == 3 and log_probs.shape[0] == 1:
                    log_probs = log_probs.squeeze(0)

                dev_loss += loss_fcn(log_probs, y).item()
            dev_loss /= len(dev)

        print(f"  Dev Loss: {dev_loss:.4f}\n")

    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(
    model: Transformer,
    dev_examples: List[LetterCountingExample],
    do_print=False,
    do_plot_attn=False,
):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print(
            "Decoding on a large number of examples (%i); not printing or plotting"
            % len(dev_examples)
        )
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(
                    attn_map.detach().numpy(), cmap="hot", interpolation="nearest"
                )
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print(
        "Accuracy: %i / %i = %f"
        % (num_correct, num_total, float(num_correct) / num_total)
    )
