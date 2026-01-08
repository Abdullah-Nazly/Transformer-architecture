# models.py

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from transformer import Transformer


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, transformer_model, vocab_index):
        self.model = transformer_model
        self.vocab_index = vocab_index
        self.device = next(self.model.parameters()).device

    def get_next_char_log_probs(self, context):
        self.model.eval()

        if len(context) == 0:
            context_idxs = [0]  # pick first vocab index as placeholder
        else:
            context_idxs = [self.vocab_index.index_of(c) for c in context]

        input_tensor = torch.tensor(
            context_idxs, dtype=torch.long, device=self.device
        )  # (seq_len,)

        with torch.no_grad():
            log_probs, _ = self.model(
                input_tensor
            )  # log_probs: (seq_len, vocab_size) when input is 1D

        next_logits = log_probs[-1, :]
        return next_logits.cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        total_log_prob = 0.0
        cur_context = context
        for ch in next_chars:
            log_probs = self.get_next_char_log_probs(cur_context)
            ch_idx = self.vocab_index.index_of(ch)
            total_log_prob += log_probs[ch_idx]
            cur_context += ch
        return float(total_log_prob)


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    seq_len = getattr(args, "num_positions", 256)
    d_model = getattr(args, "d_model", 128)
    d_internal = getattr(args, "d_internal", 256)
    num_layers = getattr(args, "num_layers", 3)
    num_heads = getattr(args, "num_heads", 4)
    lr = getattr(args, "lr", 1e-3)
    num_epochs = getattr(args, "num_epochs", 6)
    vocab_size = len(vocab_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------#
    # Data Preparation  #
    # ------------------#
    def text_to_tensor(text):
        return torch.tensor(
            [vocab_index.index_of(ch) for ch in text],
            dtype=torch.long,
        )

    train_text_limited = train_text
    dev_text_limited = dev_text

    print(
        f"Using {len(train_text_limited)} training chars, {len(dev_text_limited)} dev chars"
    )

    train_tensor = text_to_tensor(train_text_limited)
    dev_tensor = text_to_tensor(dev_text_limited)

    def make_chunks(text_tensor, seq_len, stride=None):
        if stride is None:
            stride = seq_len
        inputs, targets = [], []
        if len(text_tensor) <= seq_len:
            if len(text_tensor) > 1:
                pad_len = max(0, seq_len - len(text_tensor) + 1)
                padded = torch.cat(
                    [text_tensor, torch.zeros(pad_len, dtype=torch.long)]
                )
                inputs.append(padded[:seq_len])
                targets.append(padded[1 : seq_len + 1])
        else:
            for i in range(0, len(text_tensor) - seq_len, stride):
                inputs.append(text_tensor[i : i + seq_len])
                targets.append(text_tensor[i + 1 : i + seq_len + 1])
        if len(inputs) == 0:
            return torch.empty(0, seq_len, dtype=torch.long), torch.empty(
                0, seq_len, dtype=torch.long
            )
        return torch.stack(inputs), torch.stack(targets)

    # Use more overlap for small datasets to get more training samples
    chunk_stride = seq_len // 8 if len(train_tensor) < 200000 else seq_len // 4
    train_inputs, train_targets = make_chunks(
        train_tensor, seq_len, stride=chunk_stride
    )
    print(
        f"Training chunks: {len(train_inputs)} (from {len(train_text_limited)} chars)"
    )
    print(
        f"Sequence length: {seq_len}, Vocab size: {vocab_size}, Num heads: {num_heads}"
    )

    if len(train_inputs) == 0:
        raise ValueError(
            f"Not enough training data! Need at least {seq_len + 1} characters, got {len(train_tensor)}"
        )

    model = Transformer(
        vocab_size=vocab_size,
        num_positions=seq_len,
        d_model=d_model,
        d_internal=d_internal,
        num_classes=vocab_size,
        num_layers=num_layers,
        use_positional_encoding=True,
        use_causal_mask=True,
        batched_pe=True,
        num_heads=num_heads,
    ).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=1, min_lr=1e-5
    )

    model.train()
    dev_inputs, dev_targets = make_chunks(dev_tensor, seq_len, stride=seq_len)
    dev_sample_size = min(len(dev_inputs), 50) if len(dev_inputs) > 0 else 0

    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        t0 = time.time()

        indices = torch.randperm(len(train_inputs))
        train_inputs_shuffled = train_inputs[indices]
        train_targets_shuffled = train_targets[indices]

        chunk_count = 0
        total_chunks = len(train_inputs_shuffled)
        chunk_batch_size = 10

        for i in range(0, total_chunks, chunk_batch_size):
            batch_end = min(i + chunk_batch_size, total_chunks)
            inp = train_inputs_shuffled[i:batch_end].to(device)
            tgt = train_targets_shuffled[i:batch_end].to(device)

            optimizer.zero_grad()
            log_probs, _ = model(inp)
            loss = criterion(log_probs.view(-1, vocab_size), tgt.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            chunk_count += batch_end - i
            if chunk_count % (chunk_batch_size * 300) == 0:
                print(f"  Chunk {chunk_count}/{total_chunks} - Loss: {loss.item():.4f}")

        t1 = time.time()
        num_batches = (total_chunks + chunk_batch_size - 1) // chunk_batch_size
        avg_loss = epoch_loss / num_batches if num_batches > 0 else epoch_loss
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

        if (epoch + 1) % 1 == 0 and dev_sample_size > 0:
            model.eval()
            with torch.no_grad():
                dev_loss = 0.0
                dev_chunks = 0
                dev_batch_size = 8
                for j in range(0, dev_sample_size, dev_batch_size):
                    batch_end = min(j + dev_batch_size, dev_sample_size)
                    dev_inp = dev_inputs[j:batch_end].to(device)
                    dev_tgt = dev_targets[j:batch_end].to(device)
                    dev_log_probs, _ = model(dev_inp)
                    batch_loss = criterion(
                        dev_log_probs.view(-1, vocab_size), dev_tgt.view(-1)
                    ).item()
                    num_elements = (batch_end - j) * seq_len
                    dev_loss += batch_loss * num_elements
                    dev_chunks += batch_end - j
                if dev_chunks > 0:
                    avg_dev_loss = dev_loss / (dev_chunks * seq_len)
                    est_perplexity = np.exp(avg_dev_loss)
                    print(f"  Dev Loss: {avg_dev_loss:.4f}")
                    scheduler.step(avg_dev_loss)
            model.train()

    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    return NeuralLanguageModel(model, vocab_index)
