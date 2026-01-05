# Complete Explanation: TransformerLayer

## Overview

The `TransformerLayer` is the core building block of the transformer architecture. It processes sequences through self-attention and feed-forward networks, with residual connections and layer normalization.

---

## Input and Output

### Input
```python
input_vecs: [batch_size, seq_len, d_model]
```
- **batch_size**: Number of examples (can be 1)
- **seq_len**: Sequence length (20 in your case)
- **d_model**: Model dimension (128 in your case)

**Example:** `[1, 20, 128]` - one example, 20 positions, 128 dimensions each

### Output
```python
output: [batch_size, seq_len, d_model]  # Same shape as input!
attention_weights: [batch_size, seq_len, seq_len]  # Attention matrix
```
- **output**: Processed sequence (same shape as input)
- **attention_weights**: Attention matrix showing which positions attend to which

---

## Architecture Components

### Initialization (Lines 185-222)

#### 1. Multi-Head Setup (Lines 187-194)
```python
self.num_heads = num_heads  # Default: 1 (single-head)
self.use_multihead = num_heads > 1

if self.use_multihead:
    assert d_internal % num_heads == 0  # Must be divisible
    self.d_head = d_internal // num_heads  # Dimension per head
```

**What this does:**
- Checks if using multi-head attention
- For multi-head: splits `d_internal` (256) into `num_heads` parts
- Example: `d_internal=256`, `num_heads=4` → `d_head=64` per head

#### 2. Q, K, V Projections (Lines 197-199)
```python
self.w_queries = nn.Linear(d_model, d_internal)  # [128, 256]
self.w_keys = nn.Linear(d_model, d_internal)     # [128, 256]
self.w_values = nn.Linear(d_model, d_internal)   # [128, 256]
```

**Purpose:**
- Transform input vectors into Query, Key, Value representations
- Query: "What am I looking for?"
- Key: "What do I represent?"
- Value: "What information do I contain?"

#### 3. Output Projection (Line 202)
```python
self.w_output = nn.Linear(d_internal, d_model)  # [256, 128]
```

**Purpose:**
- Projects attention output back to `d_model` dimension
- Needed for residual connection (must match input size)

#### 4. Feed-Forward Network (Lines 205-209)
```python
self.feed_forward = nn.Sequential(
    nn.Linear(d_model, d_internal),  # Expand: [128, 256]
    ReLU(),                          # Non-linearity
    nn.Linear(d_internal, d_model),  # Contract: [256, 128]
)
```

**Purpose:**
- Processes each position independently
- Two linear layers with ReLU activation
- Expands to `d_internal`, then contracts back to `d_model`

#### 5. Layer Normalization (Lines 219-220)
```python
self.layer_norm1 = LayerNorm(d_model)  # After attention
self.layer_norm2 = LayerNorm(d_model)  # After feed-forward
```

**Purpose:**
- Normalizes activations for stable training
- Applied after each sub-layer (attention, feed-forward)

---

## Step-by-Step Forward Pass

### Step 1: Compute Q, K, V (Lines 229-231)

```python
Q = self.w_queries(input_vecs)  # [1, 20, 128] → [1, 20, 256]
K = self.w_keys(input_vecs)    # [1, 20, 128] → [1, 20, 256]
V = self.w_values(input_vecs)   # [1, 20, 128] → [1, 20, 256]
```

**What happens:**
- Each position gets projected to 256 dimensions
- Q, K, V all have shape `[batch, seq_len, d_internal]`

**Shape transformation:**
```
Input:  [1, 20, 128]  (d_model)
        ↓
Q, K, V: [1, 20, 256]  (d_internal)
```

---

### Step 2: Attention Computation

The code branches into two paths: **Multi-Head** or **Single-Head** attention.

---

## Single-Head Attention (Lines 270-289)

### Step 2a: Compute Attention Scores (Line 271)

```python
attention_score = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(K.size(-1))
```

**What this does:**
1. `K.transpose(1, 2)`: Transpose K from `[1, 20, 256]` to `[1, 256, 20]`
2. `Q @ K.T`: Matrix multiplication → `[1, 20, 20]`
3. Divide by `sqrt(256)`: Scaling factor to prevent large values

**Mathematical formula:**
```
attention_score[i, j] = (Q[i] · K[j]) / sqrt(d_internal)
```

**Result shape:** `[1, 20, 20]`
- Row `i`: attention scores for position `i` to all positions
- Column `j`: how much position `j` is attended to

**Example:**
```
attention_score[0, :] = [0.5, 0.2, 0.1, 0.05, ...]
                        ↑    ↑    ↑    ↑
                    pos0  pos1 pos2 pos3
Position 0 attends most to itself, then position 1, etc.
```

### Step 2b: Apply Causal Mask (Lines 274-283) - If Enabled

```python
if self.use_causal_mask:
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, ...),
        diagonal=1
    )
    attention_score = attention_score.masked_fill(
        causal_mask.unsqueeze(0), float("-inf")
    )
```

**What causal mask does:**
- Creates upper triangular matrix (1s above diagonal, 0s on/below)
- Masks future positions (sets to `-inf`)
- Ensures position `i` can only attend to positions `≤ i`

**Visual representation:**
```
Without mask:          With causal mask:
[0.5 0.2 0.1 0.05]    [0.5  -inf -inf -inf]
[0.3 0.4 0.2 0.1]     [0.3  0.4  -inf -inf]
[0.1 0.2 0.5 0.2]     [0.1  0.2  0.5  -inf]
[0.05 0.1 0.2 0.65]   [0.05 0.1  0.2  0.65]
```

**Why needed:**
- For autoregressive tasks (predicting next token)
- Prevents "cheating" by looking at future
- Position 0 can only see itself
- Position 1 can see positions 0-1
- Position 2 can see positions 0-2
- etc.

### Step 2c: Apply Softmax (Line 285)

```python
attention_weights = self.softmax(attention_score)
```

**What this does:**
- Converts scores to probabilities
- Each row sums to 1.0
- `-inf` values become 0.0 after softmax

**Result:** `[1, 20, 20]` - probability distribution for each position

### Step 2d: Apply Dropout (Lines 286-288)

```python
attention_weights = self.attention_dropout(attention_weights)
```

**Purpose:**
- Randomly sets some attention weights to 0 (10% chance)
- Prevents overfitting
- Only during training

### Step 2e: Apply Attention to Values (Line 289)

```python
attention_output = torch.matmul(attention_weights, V)
```

**What this does:**
- Weighted sum of V vectors using attention weights
- `[1, 20, 20] @ [1, 20, 256] = [1, 20, 256]`

**Mathematical formula:**
```
attention_output[i] = Σ(attention_weights[i, j] × V[j])
```

**Result:** `[1, 20, 256]` - attended representations

---

## Multi-Head Attention (Lines 233-268)

Multi-head attention splits the computation into multiple parallel "heads", each attending to different aspects.

### Step 2a: Split into Heads (Lines 236-238)

```python
Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
```

**What this does:**
- Reshapes `[1, 20, 256]` into `[1, 20, 4, 64]` (for 4 heads)
- Transposes to `[1, 4, 20, 64]` (heads first)

**Example with 4 heads:**
```
Before: Q = [1, 20, 256]  (one big vector per position)
After:  Q = [1, 4, 20, 64]  (4 smaller vectors per position)
        ↑   ↑  ↑   ↑
      batch heads seq dim_per_head
```

**Why split:**
- Each head can focus on different relationships
- Head 1 might focus on syntax
- Head 2 might focus on semantics
- Head 3 might focus on long-range dependencies
- Head 4 might focus on local patterns

### Step 2b: Compute Attention for Each Head (Lines 241-243)

```python
attention_score = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
```

**What this does:**
- Same as single-head, but for each head independently
- `[1, 4, 20, 64] @ [1, 4, 64, 20] = [1, 4, 20, 20]`
- Each head computes its own attention matrix

**Result:** `[1, 4, 20, 20]` - 4 attention matrices

### Step 2c: Apply Causal Mask (Lines 246-255) - If Enabled

```python
if self.use_causal_mask:
    causal_mask = torch.triu(...)  # [20, 20]
    attention_score = attention_score.masked_fill(
        causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
    )
```

**Why needed for multi-head:**
- **Same reason as single-head**: Prevent looking at future
- **Applied to each head independently**: Each head is masked
- **Shape handling**: `unsqueeze(0).unsqueeze(0)` adds batch and head dimensions
  - `[20, 20]` → `[1, 1, 20, 20]` for broadcasting

**Key point:** Causal masking is **essential** for autoregressive tasks, whether single-head or multi-head. Each head must respect causality.

### Step 2d: Softmax and Dropout (Lines 257-260)

```python
attention_weights = self.softmax(attention_score)  # [1, 4, 20, 20]
attention_weights = self.attention_dropout(attention_weights)
```

**Same as single-head**, but for each head independently.

### Step 2e: Apply to Values (Line 261)

```python
attention_output = torch.matmul(attention_weights, V)
```

**Result:** `[1, 4, 20, 64]` - attended output for each head

### Step 2f: Concatenate Heads (Lines 264-265)

```python
attention_output = attention_output.transpose(1, 2).contiguous()
attention_output = attention_output.view(batch_size, seq_len, -1)
```

**What this does:**
1. Transpose: `[1, 4, 20, 64]` → `[1, 20, 4, 64]`
2. Reshape: `[1, 20, 4, 64]` → `[1, 20, 256]` (concatenate heads)

**Result:** `[1, 20, 256]` - same as single-head output!

### Step 2g: Average Attention Weights (Line 268)

```python
attention_weights = attention_weights.mean(dim=1)
```

**Purpose:**
- Averages across heads for visualization/compatibility
- `[1, 4, 20, 20]` → `[1, 20, 20]`
- Returns single attention matrix (average of all heads)

---

## Step 3: Project Back to d_model (Line 292)

```python
attention_output = self.w_output(attention_output)
```

**What this does:**
- Projects from `d_internal` (256) back to `d_model` (128)
- `[1, 20, 256]` → `[1, 20, 128]`

**Why needed:**
- Must match input dimension for residual connection
- Input was `[1, 20, 128]`, output must be `[1, 20, 128]`

---

## Step 4: Residual Connection + Layer Norm (Line 295)

```python
x = self.layer_norm1(input_vecs + attention_output)
```

**What this does:**
1. **Residual connection**: `input_vecs + attention_output`
   - Adds original input to attention output
   - Helps with gradient flow
   - Allows model to learn identity if needed

2. **Layer normalization**: Normalizes the sum
   - Keeps values stable
   - Prevents exploding activations

**Shape:** `[1, 20, 128]` (same as input)

**Why residual connection:**
- Allows gradients to flow directly
- Model can learn: `output = input + transformation`
- Easier to train deep networks

---

## Step 5: Feed-Forward Network (Line 298)

```python
feed_forward_output = self.feed_forward(x)
```

**What this does:**
- Processes each position independently
- `[1, 20, 128]` → `[1, 20, 128]` (same shape)

**Internal flow:**
```
[1, 20, 128] → Linear → [1, 20, 256] → ReLU → [1, 20, 256] → Linear → [1, 20, 128]
```

**Purpose:**
- Adds non-linearity
- Processes information at each position
- Works independently (no interaction between positions)

---

## Step 6: Residual Connection + Layer Norm (Line 301)

```python
output = self.layer_norm2(x + feed_forward_output)
```

**What this does:**
- Same pattern as Step 4
- Residual: `x + feed_forward_output`
- Layer norm: Normalize the result

**Result:** `[1, 20, 128]` - final output

---

## Complete Flow Diagram

```
INPUT: [1, 20, 128]
       ↓
┌─────────────────────────────────────┐
│ 1. Q, K, V Projections             │
│    [1, 20, 128] → [1, 20, 256]     │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ 2. Attention Computation            │
│    Single-head: [1, 20, 256]        │
│    Multi-head:  [1, 4, 20, 64]      │
│    → Compute scores                  │
│    → Apply causal mask (if needed)   │
│    → Softmax → Dropout               │
│    → Apply to V                      │
│    Result: [1, 20, 256]             │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ 3. Output Projection                │
│    [1, 20, 256] → [1, 20, 128]     │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ 4. Residual + Layer Norm            │
│    input + attention_output         │
│    → LayerNorm                       │
│    Result: [1, 20, 128]             │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ 5. Feed-Forward Network              │
│    [1, 20, 128] → [1, 20, 256]     │
│    → ReLU → [1, 20, 256]           │
│    → [1, 20, 128]                   │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ 6. Residual + Layer Norm            │
│    x + feed_forward_output          │
│    → LayerNorm                       │
│    Result: [1, 20, 128]             │
└─────────────────────────────────────┘
       ↓
OUTPUT: [1, 20, 128]  (same as input!)
```

---

## Single-Head vs Multi-Head: Key Differences

### Single-Head Attention
- **One attention computation**
- Simpler, faster
- All relationships learned in one pass
- Shape: `[batch, seq_len, d_internal]`

### Multi-Head Attention
- **Multiple parallel attention computations**
- More expressive, can learn diverse relationships
- Each head focuses on different aspects
- Shape: `[batch, num_heads, seq_len, d_head]` → concatenated to `[batch, seq_len, d_internal]`

### When to Use Each

**Single-Head:**
- Simpler tasks
- Limited compute
- Faster training
- Your default: `num_heads=1`

**Multi-Head:**
- Complex tasks
- Need diverse attention patterns
- More parameters, slower
- Common: `num_heads=4, 8, 16`

---

## Causal Masking: Detailed Explanation

### What is Causal Masking?

Causal masking ensures that position `i` can only attend to positions `≤ i` (past and present, not future).

### Why It's Needed

**For autoregressive tasks:**
- Language modeling: Predict next token given previous tokens
- Must not "cheat" by looking at future
- Training: Model sees full sequence, but should only use past
- Inference: Model generates one token at a time (only past available)

**For your letter-counting task:**
- BEFORE task: Similar to autoregressive (only past information)
- BEFOREAFTER task: Can use full sequence (no causal mask needed)

### How It Works

```python
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
```

**Creates:**
```
[0 1 1 1]  ← Position 0 can't see future (1, 2, 3)
[0 0 1 1]  ← Position 1 can't see future (2, 3)
[0 0 0 1]  ← Position 2 can't see future (3)
[0 0 0 0]  ← Position 3 (last) can see all past
```

**Applied to attention scores:**
- Masked positions set to `-inf`
- After softmax: `-inf` → 0.0
- Result: Future positions get 0 attention weight

### Why Needed for Multi-Head

**Same reason as single-head:**
- Each head must respect causality
- All heads are masked independently
- Prevents any head from "cheating"

**Implementation difference:**
- Single-head: `mask.unsqueeze(0)` → `[1, 20, 20]`
- Multi-head: `mask.unsqueeze(0).unsqueeze(0)` → `[1, 1, 20, 20]` (broadcasts to all heads)

---

## Key Takeaways

1. **Input/Output**: Same shape `[batch, seq_len, d_model]`
2. **Two attention modes**: Single-head (simpler) or Multi-head (more expressive)
3. **Causal masking**: Prevents looking at future (needed for autoregressive tasks)
4. **Residual connections**: Help with gradient flow and training
5. **Layer normalization**: Keeps activations stable
6. **Feed-forward**: Processes each position independently
7. **Complete flow**: Attention → Residual+Norm → Feed-forward → Residual+Norm

---

## Summary

The TransformerLayer is a powerful component that:
- Uses self-attention to let positions interact
- Supports both single-head and multi-head attention
- Applies causal masking when needed (autoregressive tasks)
- Uses residual connections for better training
- Normalizes activations for stability
- Processes information through feed-forward networks
- Maintains input/output shape for stacking multiple layers

This architecture allows the model to learn complex relationships in sequences while maintaining stable, efficient training!
