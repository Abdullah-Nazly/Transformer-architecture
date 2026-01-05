# Understanding `d_internal` with Examples

## What is `d_internal`?

`d_internal` is the **internal dimension** used inside the self-attention mechanism. It's the dimension of the Query (Q), Key (K), and Value (V) vectors during attention computation.

**Key Point:** The model operates in `d_model` space (128), but expands to `d_internal` space (256) for attention computations, then contracts back to `d_model`.

---

## The Dimension Flow

### Your Model's Dimensions:
- `d_model = 128` (model dimension - where everything starts and ends)
- `d_internal = 256` (internal dimension - used during attention)

### Why Two Dimensions?

Think of it like this:
- **`d_model`**: The "standard" dimension for the model
- **`d_internal`**: A "larger workspace" for attention computations
- **Larger workspace = more capacity** to learn complex attention patterns

---

## Step-by-Step Example

Let's trace through what happens with a concrete example:

### Input to Transformer Layer

```
Input shape: [20, 128]
- 20 positions (characters in sequence)
- 128 dimensions (d_model)
```

**Example values (simplified):**
```
Position 0: [0.1, 0.2, 0.3, ..., 0.9]  (128 numbers)
Position 1: [0.2, 0.1, 0.4, ..., 0.8]  (128 numbers)
...
Position 19: [0.3, 0.2, 0.5, ..., 0.7] (128 numbers)
```

---

## Step 1: Project to `d_internal` (Expansion)

### Query, Key, Value Projections

```python
# Lines 197-199 in transformer.py
self.w_queries = nn.Linear(d_model, d_internal)  # [128, 256]
self.w_keys = nn.Linear(d_model, d_internal)     # [128, 256]
self.w_values = nn.Linear(d_model, d_internal)    # [128, 256]
```

**What happens:**
```python
Q = self.w_queries(input_vecs)  # [20, 128] → [20, 256]
K = self.w_keys(input_vecs)     # [20, 128] → [20, 256]
V = self.w_values(input_vecs)   # [20, 128] → [20, 256]
```

**Visual representation:**
```
Input:  [20, 128]  ← d_model dimension
         ↓
    Project Q: [20, 128] × [128, 256] = [20, 256]  ← d_internal dimension
    Project K: [20, 128] × [128, 256] = [20, 256]  ← d_internal dimension
    Project V: [20, 128] × [128, 256] = [20, 256]  ← d_internal dimension
```

**Why expand?**
- More dimensions = more capacity to learn complex relationships
- Attention can capture richer patterns in this larger space

---

## Step 2: Attention Computation (in `d_internal` space)

### Attention Score Calculation

```python
attention_score = Q @ K.T / sqrt(d_internal)
# [20, 256] @ [256, 20] = [20, 20]
```

**What this does:**
- Computes similarity between all position pairs
- Uses the expanded `d_internal` dimension (256) for richer comparisons
- Result: `[20, 20]` attention matrix

### Attention Output

```python
attention_output = attention_weights @ V
# [20, 20] @ [20, 256] = [20, 256]
```

**Result:** Still in `d_internal` space `[20, 256]`

---

## Step 3: Project Back to `d_model` (Contraction)

### Output Projection

```python
# Line 202 in transformer.py
self.w_output = nn.Linear(d_internal, d_model)  # [256, 128]

attention_output = self.w_output(attention_output)
# [20, 256] → [20, 128]
```

**Visual representation:**
```
Attention Output: [20, 256]  ← d_internal dimension
                  ↓
         Project: [20, 256] × [256, 128] = [20, 128]  ← back to d_model
```

**Why contract back?**
- Must match input dimension for residual connection
- Maintains consistent `d_model` dimension throughout the model

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ Input: [20, 128] (d_model)                              │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ EXPAND to d_internal                                     │
│ Q: [20, 128] → [20, 256]  (d_internal)                   │
│ K: [20, 128] → [20, 256]  (d_internal)                  │
│ V: [20, 128] → [20, 256]  (d_internal)                   │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ ATTENTION COMPUTATION (in d_internal space)              │
│ Q @ K.T: [20, 256] @ [256, 20] = [20, 20]               │
│ attention_weights @ V: [20, 20] @ [20, 256] = [20, 256] │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ CONTRACT back to d_model                                 │
│ [20, 256] → [20, 128]  (d_model)                        │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Output: [20, 128] (d_model) - same as input!             │
└─────────────────────────────────────────────────────────┘
```

---

## Why is `d_internal` Larger?

### Typical Ratio: `d_internal = 2 × d_model`

In your model:
- `d_model = 128`
- `d_internal = 256` (exactly 2x)

### Reasons:

1. **More Capacity for Attention**
   - Larger dimension = more parameters in Q, K, V projections
   - Can learn more complex attention patterns
   - Better at capturing relationships between positions

2. **Standard Practice**
   - Research shows 2-4x expansion works well
   - Common in transformer architectures (BERT, GPT, etc.)

3. **Computational Trade-off**
   - Larger = more computation, but better expressiveness
   - 2x is a good balance

---

## Feed-Forward Network Also Uses `d_internal`

Looking at lines 205-209:

```python
self.feed_forward = nn.Sequential(
    nn.Linear(d_model, d_internal),  # Expand: [128, 256]
    ReLU(),
    nn.Linear(d_internal, d_model),  # Contract: [256, 128]
)
```

**Flow:**
```
Input:  [20, 128] (d_model)
         ↓
Expand: [20, 128] → [20, 256] (d_internal)
         ↓
ReLU:   [20, 256] (non-linearity)
         ↓
Contract: [20, 256] → [20, 128] (d_model)
         ↓
Output: [20, 128] (d_model)
```

**Same pattern:** Expand → Process → Contract

---

## Concrete Numerical Example

Let's say we have one position with a simplified vector:

### Input (d_model = 128)
```
Position 0: [0.1, 0.2, 0.3, ..., 0.9]  (128 values)
```

### After Q Projection (d_internal = 256)
```
Q[0]: [0.15, 0.25, 0.35, ..., 0.95]  (256 values - expanded!)
```

### Attention Computation
```
Compare Q[0] with all K vectors (each 256-dim)
→ Get attention scores for position 0
→ Weighted sum of V vectors (each 256-dim)
→ Result: [0.12, 0.22, 0.32, ..., 0.92]  (256 values)
```

### After Output Projection (back to d_model = 128)
```
Output[0]: [0.11, 0.21, 0.31, ..., 0.91]  (128 values - contracted!)
```

---

## Parameter Count Impact

### Attention Projections:
- Q projection: `128 × 256 = 32,768` parameters
- K projection: `128 × 256 = 32,768` parameters
- V projection: `128 × 256 = 32,768` parameters
- Output projection: `256 × 128 = 32,768` parameters
- **Total: 131,072 parameters**

### If `d_internal = d_model = 128`:
- Q, K, V, Output: `4 × (128 × 128) = 65,536` parameters
- **Half the parameters!**

### Trade-off:
- Smaller `d_internal`: Fewer parameters, less capacity
- Larger `d_internal`: More parameters, more capacity
- `d_internal = 256` (2x) is a sweet spot

---

## Key Takeaways

1. **`d_internal` is the "workspace" dimension**
   - Used during attention and feed-forward computations
   - Typically 2-4x larger than `d_model`

2. **Dimension Flow:**
   - Input: `[seq_len, d_model]`
   - Expand: `[seq_len, d_internal]` (for computation)
   - Contract: `[seq_len, d_model]` (for output)

3. **Why larger?**
   - More capacity to learn complex patterns
   - Standard practice in transformers
   - Good balance between capacity and efficiency

4. **Used in two places:**
   - Self-attention (Q, K, V projections)
   - Feed-forward network (first linear layer)

5. **Must contract back:**
   - Output must match input dimension for residual connections
   - Maintains consistent `d_model` throughout the model

---

## Analogy

Think of `d_internal` like a **workshop**:
- You work in a small room (`d_model = 128`)
- But when you need to do complex work, you expand to a larger workshop (`d_internal = 256`)
- You do your work in the larger space (more room, more tools)
- Then you bring the result back to your small room

The larger workspace gives you more capacity to do complex operations!
