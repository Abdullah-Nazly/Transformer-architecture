# Complete Explanation: Transformer Class

## Overview

The `Transformer` class is the complete model architecture that processes sequences through embeddings, positional encoding, multiple transformer layers, and output classification. It's the main model used for the letter-counting task.

---

## Architecture Components

The Transformer consists of four main components:

1. **Token Embedding**: Converts integer indices to dense vectors
2. **Positional Encoding**: Adds position information to embeddings
3. **TransformerLayers**: Stack of processing layers (self-attention + feed-forward)
4. **Output Layer**: Final classification layer

---

## Input and Output

### Input

```python
indices: [batch_size, seq_len] or [seq_len]
```

**Example:** `[1, 20]` or `[20]`
- Integer indices representing characters (0-26 for vocab)
- Can be single example or batched

**In your code:**
- From `LetterCountingExample.input_tensor`
- Shape: `[20]` (single example) or `[batch, 20]` (batched)

### Output

```python
log_probs: [seq_len, num_classes] or [batch_size, seq_len, num_classes]
attention_maps: List of [seq_len, seq_len] matrices
```

**Example:** `[20, 3]` log probabilities + list of `[20, 20]` attention matrices

**What it contains:**
- `log_probs`: Log probabilities for each position and class (0, 1, or 2)
- `attention_maps`: One attention matrix per TransformerLayer (for visualization)

---

## Initialization (Lines 90-137)

### Parameters

```python
def __init__(
    self,
    vocab_size=27,              # Vocabulary size (26 letters + space)
    num_positions=20,           # Maximum sequence length
    d_model=128,                # Model dimension
    d_internal=256,             # Internal attention dimension
    num_classes=3,              # Output classes (0, 1, 2)
    num_layers=1,               # Number of transformer layers
    use_positional_encoding=True,  # Whether to add positional info
    use_causal_mask=False,      # Whether to use causal masking
    batched_pe=False,           # Whether positional encoding is batched
    num_heads=1,                # Number of attention heads
):
```

### Component Creation

#### 1. Token Embedding (Line 118)

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
```

**What it creates:**
- Lookup table: `[27, 128]`
- Maps each vocabulary index (0-26) to a 128-dimensional vector
- Learnable parameters: 27 × 128 = 3,456

**Purpose:**
- Converts sparse integer indices to dense, learnable vectors
- Each character gets its own embedding vector

#### 2. Positional Encoding (Lines 121-123)

```python
self.positional_encoding = PositionalEncoding(
    d_model, num_positions, batched=batched_pe
)
```

**What it creates:**
- Learnable embeddings for positions: `[20, 128]`
- Each position (0-19) gets its own 128-dimensional vector
- Learnable parameters: 20 × 128 = 2,560

**Purpose:**
- Adds position information to token embeddings
- Allows model to distinguish order (e.g., "hello" vs "olleh")
- Can be disabled with `use_positional_encoding=False`

#### 3. Transformer Layers (Lines 125-135)

```python
self.layers = nn.ModuleList([
    TransformerLayer(
        d_model,
        d_internal,
        use_causal_mask=use_causal_mask,
        num_heads=num_heads,
    )
    for _ in range(num_layers)
])
```

**What it creates:**
- Stack of `num_layers` TransformerLayer instances
- Each layer processes sequences through self-attention and feed-forward
- All layers share the same configuration

**Purpose:**
- Each layer refines the representations
- More layers = deeper understanding (but more parameters and computation)

#### 4. Output Layer (Line 137)

```python
self.output_layer = nn.Linear(d_model, num_classes)
```

**What it creates:**
- Linear layer: `[128, 3]`
- Maps from model dimension to number of classes
- Learnable parameters: 128 × 3 = 384

**Purpose:**
- Final classification at each position
- Outputs logits for 3 classes (0, 1, or 2 occurrences)

---

## Forward Pass: Step-by-Step (Lines 139-172)

### Step 1: Handle Batch Dimension (Lines 146-147)

```python
if indices.dim() == 1:
    indices = indices.unsqueeze(0)  # Adds batch dimension if missing
```

**What this does:**
- If input is `[20]` (single example), adds batch dimension → `[1, 20]`
- Ensures consistent shape for all operations

**Why needed:**
- Model expects batch dimension for consistency
- Makes code work with both single examples and batches

**Shape transformation:**
```
Input:  [20] or [batch, 20]
Output: [1, 20] or [batch, 20] (always has batch dimension)
```

---

### Step 2: Token Embedding (Line 150)

```python
x = self.token_embedding(indices)
```

**What this does:**
- Looks up embedding vector for each integer index
- Converts sparse indices to dense vectors

**Shape transformation:**
```
Input:  [1, 20] (integer indices)
Output: [1, 20, 128] (dense vectors)
```

**Example:**
```python
# Input indices
indices = [7, 4, 11, 11, 14, ...]  # "hello..."

# After embedding
x[0] = [0.23, -0.45, 0.67, ..., 0.12]  # 'h' embedding (128 dims)
x[1] = [0.34, 0.56, -0.23, ..., 0.45]  # 'e' embedding (128 dims)
...
```

**What each vector contains:**
- Learned representation of the character
- Captures semantic information about the character
- Same character always gets same embedding (initially)

---

### Step 3: Positional Encoding (Lines 152-153)

```python
if self.use_positional_encoding:
    x = self.positional_encoding(x)
```

**What this does:**
- Adds position information to each position's embedding
- Combines character information with position information

**Shape transformation:**
```
Input:  [1, 20, 128] (character embeddings)
Output: [1, 20, 128] (character + position embeddings, same shape!)
```

**How PositionalEncoding works:**

1. **Get sequence length:**
   ```python
   input_size = x.shape[-2]  # 20
   ```

2. **Create position indices:**
   ```python
   position_indices = [0, 1, 2, ..., 19]
   ```

3. **Look up positional embeddings:**
   ```python
   pos_emb = self.emb(position_indices)  # [20, 128]
   ```
   - Position 0 → `[128-dim vector]`
   - Position 1 → `[128-dim vector]`
   - ...
   - Position 19 → `[128-dim vector]`

4. **Add to input:**
   ```python
   x = x + pos_emb  # Element-wise addition
   ```

**Example:**
```python
# Before positional encoding
Position 0 ('h'): [0.23, -0.45, 0.67, ..., 0.12]  # Character only
Position 1 ('e'): [0.34, 0.56, -0.23, ..., 0.45]  # Character only

# Positional embeddings
Position 0: [0.10, -0.20, 0.30, ..., 0.05]  # Position 0 vector
Position 1: [0.15, -0.25, 0.35, ..., 0.08]  # Position 1 vector

# After adding (element-wise)
Position 0: [0.33, -0.65, 0.97, ..., 0.17]  # Character + Position
Position 1: [0.49, 0.31, 0.12, ..., 0.53]  # Character + Position
```

**Why positional encoding is needed:**
- **Without it:** Model can't distinguish "hello" from "olleh" (same characters, different order)
- **With it:** Each position knows where it is in the sequence
- **Learnable:** Model learns optimal position representations during training

**Batched vs Non-Batched:**
- `batched=False`: `[seq_len, d_model]` → `[seq_len, d_model]`
- `batched=True`: `[batch, seq_len, d_model]` → `[batch, seq_len, d_model]`
- Handles broadcasting correctly for both cases

---

### Step 4: Transformer Layers (Lines 158-162)

```python
attention_maps = []

for layer in self.layers:
    x, attn = layer(x)
    if attn.dim() == 3 and attn.shape[0] == 1:
        attn = attn.squeeze(0)
    attention_maps.append(attn)
```

**What this does:**
- Processes input through each TransformerLayer sequentially
- Each layer refines the representations
- Collects attention matrices for visualization

**Input to TransformerLayer:**
```python
x: [batch_size, seq_len, d_model]  # [1, 20, 128]
```

**Contains:**
- Character embeddings (from token embedding)
- Position information (from positional encoding)
- Combined into 128-dimensional vectors

**Output from TransformerLayer:**
```python
x: [batch_size, seq_len, d_model]  # [1, 20, 128] (same shape!)
attn: [batch_size, seq_len, seq_len] or [seq_len, seq_len]  # [1, 20, 20] or [20, 20]
```

**What happens inside TransformerLayer (summary):**
1. **Self-Attention**: Positions interact with each other
   - Each position can attend to all (or past) positions
   - Learns relationships between positions
   - Creates context-aware representations

2. **Feed-Forward**: Processes each position independently
   - Adds non-linearity
   - Refines individual position representations

3. **Residual Connections**: Preserves original information
   - `output = input + transformation`
   - Helps with gradient flow

4. **Layer Normalization**: Keeps activations stable
   - Normalizes values for stable training

**Stacking Multiple Layers:**

```
Layer 1:
  Input:  [1, 20, 128] (char + position info)
  Output: [1, 20, 128] (with basic attention patterns)

Layer 2:
  Input:  [1, 20, 128] (Layer 1 output)
  Output: [1, 20, 128] (with more refined patterns)

Layer N:
  Input:  [1, 20, 128] (Layer N-1 output)
  Output: [1, 20, 128] (with highly sophisticated patterns)
```

**Why multiple layers:**
- Each layer adds more sophisticated understanding
- Layer 1: Basic relationships
- Layer 2: More complex patterns
- Layer N: Highly refined representations
- More layers = more capacity, but also more parameters and computation

**Attention Maps:**
- One attention matrix per layer
- Shows which positions attend to which
- Useful for visualization and understanding model behavior
- Shape: `[seq_len, seq_len]` = `[20, 20]`

---

### Step 5: Output Projection (Line 165)

```python
logits = self.output_layer(x)
```

**What this does:**
- Projects from model dimension (128) to number of classes (3)
- Creates logits (raw scores) for each class at each position

**Shape transformation:**
```
Input:  [1, 20, 128] (processed representations)
Output: [1, 20, 3] (logits for 3 classes)
```

**What logits represent:**
- For each position, 3 logit values:
  - Class 0: Hasn't appeared (or 0 times elsewhere)
  - Class 1: Appeared once (before or elsewhere)
  - Class 2: Appeared 2+ times (before or elsewhere)

**Example:**
```python
# Position 0 ('h')
logits[0] = [2.3, -1.5, 0.8]  # Raw scores for classes 0, 1, 2

# Position 3 ('l' - second 'l')
logits[3] = [-0.5, 3.2, 1.1]  # Higher score for class 1 (appeared once)
```

---

### Step 6: Log Softmax (Line 166)

```python
log_probs = nn.functional.log_softmax(logits, dim=-1)
```

**What this does:**
- Converts logits to log probabilities
- Applies softmax, then takes logarithm
- Each row sums to log(1.0) = 0.0 (in log space)

**Shape transformation:**
```
Input:  [1, 20, 3] (logits)
Output: [1, 20, 3] (log probabilities, same shape)
```

**Mathematical formula:**
```
log_prob[i, j] = log(exp(logit[i, j]) / Σ(exp(logit[i, k])))
```

**Why log probabilities:**
- More numerically stable than regular probabilities
- Used with `NLLLoss` for training
- Log probabilities are negative (closer to 0 = more confident)

**Example:**
```python
# Logits
logits[0] = [2.3, -1.5, 0.8]

# After log_softmax
log_probs[0] = [-0.1, -4.9, -1.6]  # Log probabilities
# Highest value (-0.1) corresponds to class 0 (most likely)
```

---

### Step 7: Flatten (Lines 169-170)

```python
if log_probs.shape[0] == 1:
    log_probs = log_probs.squeeze(0)
```

**What this does:**
- Removes batch dimension if batch size is 1
- Makes output compatible with single-example inference

**Shape transformation:**
```
Input:  [1, 20, 3] (with batch dimension)
Output: [20, 3] (without batch dimension)
```

**Why needed:**
- For single-example inference, batch dimension is unnecessary
- Makes output shape consistent with expected format
- `[20, 3]` is easier to work with than `[1, 20, 3]` for single examples

---

## Complete Data Flow

```
Input String: "hello world        "
        ↓
Indexer: [7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3, ...]
        Shape: [20]
        ↓
Add Batch: [1, 20]
        ↓
┌─────────────────────────────────────┐
│ Token Embedding                     │
│ [1, 20] → [1, 20, 128]             │
│ Each integer → 128-dim vector       │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Positional Encoding                  │
│ [1, 20, 128] → [1, 20, 128]        │
│ Adds position info to each vector   │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ TransformerLayer 1                  │
│ Input:  [1, 20, 128]                │
│ Output: [1, 20, 128] + [20, 20]    │
│ (Self-attention + Feed-forward)     │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ TransformerLayer 2 (if num_layers>1) │
│ Input:  [1, 20, 128]                │
│ Output: [1, 20, 128] + [20, 20]    │
│ (More refined representations)      │
└─────────────────────────────────────┘
        ↓
... (more layers if num_layers > 2)
        ↓
┌─────────────────────────────────────┐
│ Output Layer                         │
│ [1, 20, 128] → [1, 20, 3]          │
│ Each position → 3 logits             │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Log Softmax                          │
│ [1, 20, 3] → [1, 20, 3]            │
│ Logits → Log probabilities           │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Flatten                              │
│ [1, 20, 3] → [20, 3]               │
│ Remove batch dimension               │
└─────────────────────────────────────┘
        ↓
Output: [20, 3] log probabilities
        + List of [20, 20] attention maps
```

---

## Positional Encoding: Deep Dive

### Why It's Needed

**Problem without positional encoding:**
- Token embeddings only contain character information
- Model can't distinguish "hello" from "olleh" (same characters, different order)
- Attention mechanism needs position information to understand sequence structure

**Solution:**
- Add position-specific information to each embedding
- Each position gets a unique "position signature"
- Model can now understand order and relationships

### Implementation Details

**PositionalEncoding class (Lines 306-339):**

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched=False):
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched
```

**What it creates:**
- Embedding layer: `nn.Embedding(20, 128)`
- Lookup table: `[20, 128]`
- Each position (0-19) has its own 128-dim vector
- These vectors are **learnable** (updated during training)

**Forward pass:**

```python
def forward(self, x):
    input_size = x.shape[-2]  # Get sequence length (20)

    # Create position indices
    position_indices = torch.arange(input_size, dtype=torch.long)
    # [0, 1, 2, ..., 19]

    # Look up positional embeddings
    pos_emb = self.emb(position_indices)  # [20, 128]

    # Add to input (element-wise)
    return x + pos_emb  # [1, 20, 128] + [20, 128] = [1, 20, 128]
```

**Batched handling:**
- `batched=False`: Direct addition (broadcasting works)
- `batched=True`: Adds batch dimension for proper broadcasting

### Learnable vs Fixed Positional Encoding

**Your implementation uses learnable embeddings:**
- Position vectors are learned during training
- Model learns optimal position representations
- More flexible than fixed encodings

**Alternative (original Transformer paper):**
- Uses fixed sinusoidal functions
- Not learnable, but works for any sequence length
- Your approach works well for fixed-length sequences (20 chars)

### Example Walkthrough

```python
# Input after token embedding
x = [
    [0.23, -0.45, 0.67, ..., 0.12],  # Position 0 ('h')
    [0.34, 0.56, -0.23, ..., 0.45],  # Position 1 ('e')
    [0.12, -0.67, 0.89, ..., -0.34], # Position 2 ('l')
    ...
]

# Positional embeddings (learned)
pos_emb = [
    [0.10, -0.20, 0.30, ..., 0.05],  # Position 0 embedding
    [0.15, -0.25, 0.35, ..., 0.08],  # Position 1 embedding
    [0.20, -0.30, 0.40, ..., 0.11],  # Position 2 embedding
    ...
]

# After addition (element-wise)
x_with_pos = [
    [0.33, -0.65, 0.97, ..., 0.17],  # 'h' at position 0
    [0.49, 0.31, 0.12, ..., 0.53],   # 'e' at position 1
    [0.32, -0.97, 1.29, ..., -0.23], # 'l' at position 2
    ...
]
```

**Key point:** Each vector now contains both character information and position information, allowing the model to understand both what characters are and where they are.

---

## TransformerLayer Input/Output Summary

### What Goes Into TransformerLayer

```python
Input: [batch_size, seq_len, d_model]  # [1, 20, 128]
```

**Contains:**
- Character embeddings (from token embedding)
- Position information (from positional encoding)
- Combined into 128-dimensional vectors

**Each position's vector represents:**
- What character it is (semantic information)
- Where it is in the sequence (positional information)

### What Comes Out Of TransformerLayer

```python
Output: [batch_size, seq_len, d_model]  # [1, 20, 128]
Attention: [batch_size, seq_len, seq_len] or [seq_len, seq_len]  # [1, 20, 20] or [20, 20]
```

**Output contains:**
- Richer representations with context from other positions
- Self-attention has allowed positions to interact
- Feed-forward has processed each position
- Residual connections have preserved original information

**Attention matrix shows:**
- Which positions attend to which
- How much attention each position pays to others
- Useful for understanding model behavior

**Key transformation:**
- Input: Character + position information
- Output: Character + position + contextual information
- Same shape, but richer content!

---

## Key Components Summary

| Component | Input Shape | Output Shape | Parameters | Purpose |
|-----------|-------------|--------------|------------|---------|
| **Token Embedding** | `[batch, seq_len]` | `[batch, seq_len, d_model]` | 27 × 128 = 3,456 | Convert indices to vectors |
| **Positional Encoding** | `[batch, seq_len, d_model]` | `[batch, seq_len, d_model]` | 20 × 128 = 2,560 | Add position info |
| **TransformerLayer** | `[batch, seq_len, d_model]` | `[batch, seq_len, d_model]` | ~200K per layer | Process with attention |
| **Output Layer** | `[batch, seq_len, d_model]` | `[batch, seq_len, num_classes]` | 128 × 3 = 384 | Classify at each position |

---

## Key Takeaways

1. **Token Embedding**: Converts sparse integer indices to dense, learnable vectors
2. **Positional Encoding**: Adds position information so model understands order
3. **TransformerLayers**: Process sequences through self-attention and feed-forward
4. **Output Layer**: Classifies at each position into 3 classes
5. **Shape Consistency**: Input/output shapes are maintained throughout (except final classification)
6. **Stacking**: Multiple layers create increasingly sophisticated representations
7. **Attention Maps**: Provide insight into which positions the model focuses on

---

## Summary

The Transformer class is a complete sequence processing model that:

1. **Embeds** characters as dense vectors
2. **Encodes** position information
3. **Processes** through multiple transformer layers (self-attention + feed-forward)
4. **Classifies** at each position

Each component plays a crucial role:
- **Embeddings**: Represent what characters are
- **Positional encoding**: Represent where characters are
- **Transformer layers**: Represent how characters relate
- **Output layer**: Predict what to output

The architecture allows the model to understand sequences holistically while making predictions at each position, making it powerful for tasks like letter-counting where context matters!
