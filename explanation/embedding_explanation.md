# Understanding nn.Embedding in PyTorch

## What is nn.Embedding?

`nn.Embedding` is a **lookup table** that maps integer indices to dense vectors. Think of it as a dictionary where:
- **Keys**: Integer indices (0, 1, 2, ..., vocab_size-1)
- **Values**: Dense vectors of size `embedding_dim`

## Parameters

```python
nn.Embedding(vocab_size, embedding_dim)
```

### 1. `vocab_size` (num_embeddings)
- **What it is**: The number of unique tokens in your vocabulary
- **In your code**: `vocab_size = 27` (26 letters + 1 space)
- **Creates**: A lookup table with 27 rows

### 2. `embedding_dim` (d_model)
- **What it is**: The dimension of each embedding vector
- **In your code**: `d_model = 128` (default)
- **Creates**: Each row in the lookup table has 128 values

## How It Works Internally

### The Embedding Matrix

When you create:
```python
self.token_embedding = nn.Embedding(vocab_size=27, d_model=128)
```

PyTorch creates a **learnable weight matrix**:
```
Shape: [27, 128]

Row 0:  [w₀₀, w₀₁, w₀₂, ..., w₀₁₂₇]  ← embedding for index 0 ('a')
Row 1:  [w₁₀, w₁₁, w₁₂, ..., w₁₁₂₇]  ← embedding for index 1 ('b')
Row 2:  [w₂₀, w₂₁, w₂₂, ..., w₂₁₂₇]  ← embedding for index 2 ('c')
...
Row 25: [w₂₅₀, w₂₅₁, ..., w₂₅₁₂₇]    ← embedding for index 25 ('z')
Row 26: [w₂₆₀, w₂₆₁, ..., w₂₆₁₂₇]    ← embedding for index 26 (' ')
```

### Visual Diagram

```
INPUT:  Integer indices
        [7, 4, 11, 11, 14]  ← "hello"
         │  │   │   │   │
         │  │   │   │   └─→ Look up row 14 → [128-dim vector for 'o']
         │  │   │   └─────→ Look up row 11 → [128-dim vector for 'l']
         │  │   └─────────→ Look up row 11 → [128-dim vector for 'l']
         │  └─────────────→ Look up row 4  → [128-dim vector for 'e']
         └────────────────→ Look up row 7  → [128-dim vector for 'h']

OUTPUT: Dense vectors
        [
          [0.23, -0.45, 0.67, ..., 0.12],  ← 'h' embedding
          [0.34, 0.56, -0.23, ..., 0.45],  ← 'e' embedding
          [0.12, -0.67, 0.89, ..., -0.34], ← 'l' embedding
          [0.12, -0.67, 0.89, ..., -0.34], ← 'l' embedding (same as above)
          [0.45, 0.23, -0.12, ..., 0.67]   ← 'o' embedding
        ]
        Shape: [5, 128]
```

### The Forward Pass (Lookup Operation)

When you call:
```python
indices = torch.LongTensor([7, 4, 11, 11, 14])  # "hello" → [h, e, l, l, o]
x = self.token_embedding(indices)
```

**What happens**:
1. Input: `[7, 4, 11, 11, 14]` (shape: `[5]`)
2. For each index, look up the corresponding row in the embedding matrix
3. Output: `[[row 7], [row 4], [row 11], [row 11], [row 14]]` (shape: `[5, 128]`)

**Mathematically**:
```
x[i] = embedding.weight[indices[i]]
```

## Example Walkthrough

### Step 1: Character to Index (using Indexer)
```python
text = "hello"
# Using vocab_index from letter_counting.py:
indices = [7, 4, 11, 11, 14]  # h→7, e→4, l→11, l→11, o→14
```

### Step 2: Index to Tensor
```python
input_tensor = torch.LongTensor(indices)  # Shape: [5]
```

### Step 3: Embedding Lookup
```python
embedded = token_embedding(input_tensor)  # Shape: [5, 128]
```

**What this does**:
- Takes index `7` → looks up row 7 → returns 128-dim vector
- Takes index `4` → looks up row 4 → returns 128-dim vector
- Takes index `11` → looks up row 11 → returns 128-dim vector
- ... and so on

### Step 4: Result
Each character is now represented as a dense 128-dimensional vector that the transformer can process.

## Shape Transformations

```
Input:  [seq_len]           → e.g., [20] for 20 characters
Output: [seq_len, d_model]  → e.g., [20, 128] for 20 vectors of size 128
```

In your transformer code (line 149):
```python
x = self.token_embedding(indices)  # indices: [batch, 20] → x: [batch, 20, 128]
```

## Learnable Parameters

### Parameter Count
```python
Total parameters = vocab_size × embedding_dim
                 = 27 × 128
                 = 3,456 parameters
```

### How They're Learned

1. **Initialization**: Random values (usually from a normal distribution)
2. **During Training**:
   - Forward pass: Look up embeddings
   - Backward pass: Compute gradients
   - Update: Adjust embedding vectors via gradient descent
3. **After Training**: Each character has a learned representation that captures semantic relationships

### Why This Matters

- **One-hot encoding** would be: `[1, 0, 0, ..., 0]` (27-dim, sparse, no learning)
- **Embeddings** are: `[0.23, -0.45, 0.67, ...]` (128-dim, dense, learnable)

The model learns that similar characters (like 'a' and 'e', both vowels) might have similar embeddings.

## In Your Transformer Model

Looking at `transformer.py`:

```python
# Line 117: Create embedding layer
self.token_embedding = nn.Embedding(vocab_size, d_model)

# Line 149: Use it in forward pass
x = self.token_embedding(indices)  # Convert indices → dense vectors
```

**Flow**:
1. Input string: `"hello world"`
2. Indexer converts to: `[7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3]`
3. Embedding converts to: `[11, 128]` tensor (11 vectors of 128 dimensions)
4. These vectors are fed into the transformer layers

## Key Takeaways

1. **Lookup Table**: `nn.Embedding` is essentially a lookup table
2. **Learnable**: The vectors are parameters that get updated during training
3. **Dense Representation**: Converts sparse integer indices to dense vectors
4. **Shape Change**: `[seq_len]` → `[seq_len, embedding_dim]`
5. **Purpose**: Provides a meaningful numerical representation that the neural network can process

## Comparison with One-Hot Encoding

| Feature | One-Hot | Embedding |
|---------|---------|-----------|
| Dimension | vocab_size (27) | embedding_dim (128) |
| Sparsity | Sparse (mostly zeros) | Dense (all values) |
| Learnable | No | Yes |
| Memory | Fixed | Learnable |
| Semantic Info | None | Learned relationships |

