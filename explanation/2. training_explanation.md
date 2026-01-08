# Deep Explanation: Training Workflow and Hyperparameters

## Overview

The `train_classifier` function is the main training loop that takes prepared data bundles and trains a Transformer model to perform the letter-counting task.

---

## Function Signature

```python
def train_classifier(args, train, dev):
```

**Parameters:**
- `args`: Command-line arguments (contains hyperparameters)
- `train`: List of `LetterCountingExample` objects (training data)
- `dev`: List of `LetterCountingExample` objects (validation data)

**Returns:**
- Trained `Transformer` model

---

## Training Workflow Overview

```
1. Set up training configuration (batching, hyperparameters)
2. Initialize model with hyperparameters
3. Set up optimizer and loss function
4. Training loop (for each epoch):
   a. Shuffle training examples
   b. Process examples (single or batched)
   c. Forward pass → Compute loss → Backward pass → Update weights
   d. Evaluate on dev set
5. Return trained model
```

---

## Section 1: Training Configuration (Lines 351-354)

```python
# --- Enable batching here --- #
use_batched_training = False
batch_size = 16
# ----------------------------- #
```

**What this does:**
- **`use_batched_training`**: Boolean flag to enable/disable batch processing
  - `False`: Process one example at a time (default)
  - `True`: Process multiple examples together (faster, but requires padding)
- **`batch_size`**: Number of examples processed together (only used if batching enabled)

**Why it matters:**
- Single-example mode: Simpler, no padding needed
- Batched mode: More efficient GPU usage, but requires handling variable-length sequences

---

## Section 2: Hyperparameters (Lines 356-366)

### Understanding `getattr()`

```python
vocab_size = getattr(args, "vocab_size", 27)
```

**What `getattr()` does:**
- Checks if `args` has attribute `vocab_size`
- If yes: Use that value
- If no: Use default value (27)

**Why this pattern:**
- Allows command-line arguments to override defaults
- Provides sensible defaults if arguments not specified
- Makes code flexible and configurable

---

### Hyperparameter Breakdown

#### 1. `vocab_size = 27` (Line 357)

**What it is:**
- Size of the vocabulary
- Number of unique characters the model can process

**In your task:**
- 26 lowercase letters (a-z) + 1 space = 27 total

**Used for:**
- `nn.Embedding(vocab_size, d_model)` - Creates embedding lookup table with 27 rows

**Why it matters:**
- Must match the vocabulary created in `letter_counting.py`
- Each character gets its own embedding vector

---

#### 2. `num_positions = 20` (Line 358)

**What it is:**
- Maximum sequence length the model can handle
- Maximum number of positions in the input

**In your task:**
- Each example is exactly 20 characters long

**Used for:**
- `PositionalEncoding(d_model, num_positions)` - Creates positional embeddings for 20 positions
- Padding length in batched mode

**Why it matters:**
- Determines how many positional encodings to pre-compute
- Must be >= actual sequence length

---

#### 3. `d_model = 128` (Line 359)

**What it is:**
- **Model dimension** - The size of embedding vectors and hidden states
- Dimension of the vector space the model operates in

**Used for:**
- Embedding dimension: `nn.Embedding(vocab_size, d_model)`
- Input/output dimension of transformer layers
- Positional encoding dimension

**Why it matters:**
- Larger = more capacity, but more parameters and slower
- Smaller = faster, but less expressive
- 128 is a common choice for smaller models

**Shape implications:**
- Input: `[seq_len]` → After embedding: `[seq_len, d_model]`
- All transformer operations work in this `d_model`-dimensional space

---

#### 4. `d_internal = 256` (Line 360)

**What it is:**
- **Internal dimension** - Size of the attention mechanism's key/query/value vectors
- Dimension used inside the self-attention computation

**Used for:**
- Query (Q), Key (K), Value (V) projections in attention
- Feed-forward network intermediate dimension

**Why it matters:**
- Typically 2-4x larger than `d_model` for better expressiveness
- Larger = more parameters in attention, more computation
- 256 = 2x `d_model` (common ratio)

**Relationship:**
- `d_model` (128) → Project to → `d_internal` (256) → Project back to → `d_model` (128)

---

#### 5. `num_classes = 3` (Line 361)

**What it is:**
- Number of output classes for classification
- Size of the output layer

**In your task:**
- 0 = character hasn't appeared before (or 0 times elsewhere)
- 1 = character appeared once before (or 1 time elsewhere)
- 2 = character appeared 2+ times before (or 2+ times elsewhere)

**Used for:**
- `nn.Linear(d_model, num_classes)` - Final output layer
- Output shape: `[seq_len, num_classes]` (log probabilities for each class)

**Why it matters:**
- Must match the task's output space
- Each position predicts one of 3 classes

---

#### 6. `num_layers = 1` (Line 362)

**What it is:**
- Number of transformer layers (blocks) to stack
- Depth of the model

**Used for:**
- Creates `num_layers` transformer layers in sequence
- Each layer processes the input and passes to next

**Why it matters:**
- More layers = deeper model = more capacity, but slower
- 1 layer = shallow model (good for simple tasks)
- Deeper models can learn more complex patterns

**Architecture:**
```
Input → Layer 1 → Layer 2 → ... → Layer N → Output
```

---

#### 7. `lr = 1e-4` (Line 363)

**What it is:**
- **Learning rate** - Step size for gradient descent
- Controls how much weights are updated per step

**Value:**
- `1e-4` = 0.0001 (very small)
- Conservative learning rate

**Why it matters:**
- Too high = unstable training, might overshoot optimal weights
- Too low = slow convergence, might get stuck
- 1e-4 is a common default for transformers

**Used for:**
- `optim.Adam(model.parameters(), lr=lr)` - Optimizer step size

---

#### 8. `num_epochs = 10` (Line 364)

**What it is:**
- Number of complete passes through the training dataset
- How many times the model sees all training examples

**Why it matters:**
- More epochs = more training, but risk of overfitting
- 10 is a reasonable starting point
- Can be increased if model needs more training

---

#### 9. `seed = 1234` (Line 365)

**What it is:**
- Random seed for reproducibility
- Ensures same random initialization and shuffling

**Used for:**
- `torch.manual_seed(seed)` - PyTorch random number generator
- `random.seed(seed)` - Python random module
- `np.random.seed(seed)` - NumPy random module

**Why it matters:**
- Makes experiments reproducible
- Same seed → same initial weights → same results (if deterministic)

---

#### 10. `use_cuda = False` (Line 366)

**What it is:**
- Whether to use GPU (CUDA) for training
- `False` = use CPU, `True` = use GPU if available

**Used for:**
- `torch.device("cuda" if ... else "cpu")` - Selects computation device
- `.to(device)` - Moves model and data to selected device

**Why it matters:**
- GPU = much faster training (10-100x speedup)
- CPU = slower but works on any machine

---

## Section 3: Device and Random Seed Setup (Lines 368-371)

```python
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
```

**What this does:**
1. **Device selection:**
   - Checks if CUDA is requested AND available
   - Falls back to CPU if not
   - Ensures code works even without GPU

2. **Reproducibility:**
   - Sets random seeds for all random number generators
   - Ensures consistent initialization and data shuffling

**Why it matters:**
- Reproducible experiments
- Same results across runs (important for debugging and comparison)

---

## Section 4: Model Initialization (Lines 373-382)

```python
# --- Model --- #
model = Transformer(
    vocab_size=vocab_size,        # 27
    num_positions=num_positions,  # 20
    d_model=d_model,              # 128
    d_internal=d_internal,        # 256
    num_classes=num_classes,      # 3
    num_layers=num_layers,        # 1
    use_positional_encoding=True,
).to(device)
```

### What Gets Created

When you instantiate `Transformer`, it creates:

1. **Token Embedding Layer** (line 117 in Transformer.__init__):
   ```python
   self.token_embedding = nn.Embedding(vocab_size, d_model)
   # Shape: [27, 128] - Lookup table for character embeddings
   ```

2. **Positional Encoding** (lines 120-122):
   ```python
   self.positional_encoding = PositionalEncoding(d_model, num_positions, batched=False)
   # Creates positional embeddings for 20 positions, each 128-dimensional
   ```

3. **Transformer Layers** (lines 124-134):
   ```python
   self.layers = nn.ModuleList([
       TransformerLayer(d_model, d_internal, ...)
       for _ in range(num_layers)  # 1 layer
   ])
   # Creates 1 transformer layer
   ```

4. **Output Layer** (line 136):
   ```python
   self.output_layer = nn.Linear(d_model, num_classes)
   # Shape: [128, 3] - Maps from model dimension to 3 classes
   ```

### Parameter Count (Approximate)

- Embedding: `27 × 128 = 3,456`
- Positional Encoding: `20 × 128 = 2,560` (learnable embeddings)
- Transformer Layer: ~`(128 × 256) × 4 + (128 × 256) × 2 ≈ 196,608` (Q, K, V, output, FFN)
- Output Layer: `128 × 3 = 384`
- **Total: ~200,000+ parameters**

### `.to(device)` - Moving to Device

```python
model = Transformer(...).to(device)
```

**What it does:**
- Moves all model parameters to the specified device (CPU or GPU)
- All computations will happen on that device

**Why it matters:**
- GPU training is much faster
- Must match where your data tensors are located

---

## Section 5: Additional Setup (Lines 384-388)

```python
# Make positional encoding match training mode
model.positional_encoding.batched = use_batched_training

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
loss_fcn = nn.NLLLoss()
```

### Positional Encoding Configuration

- Sets whether positional encoding should handle batched inputs
- Must match the training mode (batched vs single-example)

### Optimizer: Adam

**Adam Optimizer:**
- Adaptive learning rate optimizer
- Good default choice for transformers
- `weight_decay=1e-5`: L2 regularization to prevent overfitting

**What it does:**
- Maintains per-parameter learning rates
- Adapts step size based on gradient history
- Updates all model parameters during training

### Loss Function: NLLLoss

**Negative Log Likelihood Loss:**
- Used for classification tasks
- Expects log probabilities as input (from `log_softmax`)

**Why NLLLoss:**
- Standard for multi-class classification
- Works with the model's `log_softmax` output
- Computes: `loss = -log(probability of correct class)`

---

## Complete Hyperparameter Summary Table

| Hyperparameter | Value | Purpose | Used In |
|----------------|-------|---------|---------|
| `vocab_size` | 27 | Vocabulary size | Embedding layer |
| `num_positions` | 20 | Max sequence length | Positional encoding |
| `d_model` | 128 | Model/hidden dimension | All layers |
| `d_internal` | 256 | Attention dimension | Transformer layers |
| `num_classes` | 3 | Output classes | Output layer |
| `num_layers` | 1 | Model depth | Number of transformer blocks |
| `lr` | 1e-4 | Learning rate | Optimizer |
| `num_epochs` | 10 | Training iterations | Training loop |
| `seed` | 1234 | Random seed | Reproducibility |
| `use_cuda` | False | GPU usage | Device selection |

---

## Data Flow Through Model

```
Input String: "hello world        "
    ↓
Indexer: [7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3, ...]
    ↓
Token Embedding (vocab_size=27, d_model=128):
    [20] → [20, 128]
    ↓
Positional Encoding (num_positions=20, d_model=128):
    [20, 128] → [20, 128] (adds position info)
    ↓
Transformer Layer (d_model=128, d_internal=256):
    [20, 128] → [20, 128] (self-attention + FFN)
    ↓
Output Layer (d_model=128, num_classes=3):
    [20, 128] → [20, 3] (log probabilities)
    ↓
Predictions: [20, 3] (3 log probs for each of 20 positions)
```

---

## Key Takeaways

1. **Hyperparameters control model capacity and training behavior**
2. **`d_model`** is the core dimension - everything operates in this space
3. **`d_internal`** is typically larger for attention mechanisms
4. **Model architecture** is determined by these hyperparameters
5. **Device selection** affects training speed (GPU vs CPU)
6. **Reproducibility** is ensured through random seeds

---

## Next Steps

After understanding hyperparameters, we'll dive into:
1. **Transformer Architecture** - How the model processes sequences
2. **Positional Encoding** - How position information is added
3. **Self-Attention** - The core mechanism of transformers
4. **Training Loop** - How the model learns from data

