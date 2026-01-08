# Deep Explanation: Data Preparation (lines 90-106)

## Overview

This section of the code prepares the training and development datasets for the letter-counting task. It handles two different task types: **BEFORE** and **BEFOREAFTER**, which differ in how they count letter occurrences.

---

## Line-by-Line Breakdown

### Line 90: Task Selection Logic

```python
count_only_previous = True if args.task == "BEFORE" else False
```

**What it does:**
- Sets a boolean flag based on the command-line argument `--task`
- **BEFORE**: `count_only_previous = True` (default)
- **BEFOREAFTER**: `count_only_previous = False`

**Purpose:**
- Determines the counting strategy used in `get_letter_count_output()`
- This flag is passed to the function to generate different labels

---

## Understanding the Two Tasks

### Task 1: BEFORE (count_only_previous = True)

**Counting Strategy:** Count only letters that appeared **before** the current position.

**Example:** `"hello world"`

| Position | Character | Previous Characters | Count | Output |
|----------|-----------|---------------------|-------|--------|
| 0 | h | (none) | 0 | 0 |
| 1 | e | h | 0 | 0 |
| 2 | l | he | 0 | 0 |
| 3 | l | hel | 1 (one 'l' before) | 1 |
| 4 | o | hell | 0 | 0 |
| 5 | (space) | hello | 0 | 0 |
| 6 | w | hello (space) | 0 | 0 |
| 7 | o | hello (space)w | 1 (one 'o' before) | 1 |
| 8 | r | hello (space)wo | 0 | 0 |
| 9 | l | hello (space)wor | 2 (two 'l's before) | 2 |
| 10 | d | hello (space)worl | 0 | 0 |

**Output:** `[0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0]`

**Characteristics:**
- **Causal/Autoregressive**: Only uses past information
- **Similar to language modeling**: Predicting based on previous context
- **More challenging**: Model must remember what it has seen

### Task 2: BEFOREAFTER (count_only_previous = False)

**Counting Strategy:** Count **all** occurrences of the letter in the entire string (excluding the current position).

**Example:** `"hello world"`

| Position | Character | All Other Occurrences | Count | Output |
|----------|-----------|-----------------------|-------|--------|
| 0 | h | ello world | 0 | 0 |
| 1 | e | hllo world | 0 | 0 |
| 2 | l | helo world | 2 (two 'l's elsewhere) | 2 |
| 3 | l | helo world | 2 (two 'l's elsewhere) | 2 |
| 4 | o | hell world | 1 (one 'o' elsewhere) | 1 |
| 5 | (space) | helloworld | 0 | 0 |
| 6 | w | hello orld | 0 | 0 |
| 7 | o | hello wrld | 1 (one 'o' elsewhere) | 1 |
| 8 | r | hello wold | 0 | 0 |
| 9 | l | hello word | 2 (two 'l's elsewhere) | 2 |
| 10 | d | hello worl | 0 | 0 |

**Output:** `[0, 0, 2, 2, 1, 0, 0, 1, 0, 2, 0]`

**Characteristics:**
- **Non-causal/Bidirectional**: Can use information from entire sequence
- **More information available**: Model sees the full context
- **Potentially easier**: More clues to solve the task

---

## The `get_letter_count_output()` Function

Located at lines 61-75, this function generates the labels for each position.

### Code Breakdown:

```python
def get_letter_count_output(input: str, count_only_previous: bool = True) -> np.array:
    output = np.zeros(len(input))  # Initialize output array
    for i in range(0, len(input)):
        if count_only_previous:
            # BEFORE: Count only in input[0:i] (previous characters)
            output[i] = min(2, len([c for c in input[0:i] if c == input[i]]))
        else:
            # BEFOREAFTER: Count all occurrences minus current position
            output[i] = min(2, len([c for c in input if c == input[i]]) - 1)
    return output
```

**Key Points:**
- Output values are **capped at 2** (using `min(2, count)`)
- Returns a numpy array of 0s, 1s, and 2s
- Each position gets a label indicating how many times the character appeared

---

## Data Preparation Process (Lines 92-106)

### Step 1: Read Training Examples (Line 93)

```python
train_exs = read_examples(args.train)
```

**What `read_examples()` does:**
- Opens the training file (default: `data/lettercounting-train.txt`)
- Reads each line (removes newline character)
- Each line is exactly **20 characters long**
- Returns a list of strings

**Example output:**
```python
train_exs = [
    "hello world        ",
    "the quick brown fox",
    "abcdefghijklmnopqrs",
    ...
]
```

### Step 2: Create Training Bundles (Lines 94-99)

```python
train_bundles = [
    LetterCountingExample(
        l,  # input string
        get_letter_count_output(l, count_only_previous),  # labels
        vocab_index  # for character-to-index conversion
    )
    for l in train_exs
]
```

**What happens for each example:**

1. **Input string** (`l`): The 20-character string from the file
2. **Labels** (`get_letter_count_output(...)`):
   - Generates output array based on task type
   - Array of length 20 with values 0, 1, or 2
3. **Vocab index** (`vocab_index`): Used to convert characters to integers

**Example:**
```python
# Input string
l = "hello world        "

# Generate labels (BEFORE task)
labels = get_letter_count_output(l, count_only_previous=True)
# labels = [0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Create LetterCountingExample
example = LetterCountingExample(l, labels, vocab_index)
```

### Step 3: What `LetterCountingExample` Does

Looking at `transformer.py` lines 15-21:

```python
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input  # Original string
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output  # Labels array
        self.output_tensor = torch.LongTensor(self.output)
```

**What it creates:**

1. **`self.input`**: Original string (e.g., `"hello world        "`)
2. **`self.input_indexed`**: Character indices
   - Example: `[7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3, ...]`
   - Each character converted to its index in vocabulary
3. **`self.input_tensor`**: PyTorch tensor for model input
   - Shape: `[20]` (20 integer indices)
4. **`self.output`**: Label array (0s, 1s, 2s)
   - Example: `[0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, ...]`
5. **`self.output_tensor`**: PyTorch tensor for model targets
   - Shape: `[20]` (20 labels)

### Step 4: Repeat for Dev Set (Lines 100-106)

The same process is repeated for the development set:
- Reads from `args.dev` (default: `data/lettercounting-dev.txt`)
- Creates `dev_bundles` with the same structure

---

## Complete Data Flow Example

Let's trace through a complete example:

### Input File Line:
```
"hello world        "
```

### Step 1: Task Selection
```python
args.task = "BEFOREAFTER"  # User specified
count_only_previous = False
```

### Step 2: Generate Labels
```python
labels = get_letter_count_output("hello world        ", count_only_previous=False)
# Result: [0, 0, 2, 2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

### Step 3: Create LetterCountingExample
```python
example = LetterCountingExample(
    "hello world        ",
    [0, 0, 2, 2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    vocab_index
)
```

### Step 4: Inside LetterCountingExample
```python
# Character to index conversion
input_indexed = [7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3, 26, 26, 26, 26, 26, 26, 26, 26, 26]
# h=7, e=4, l=11, l=11, o=14, space=26, w=22, o=14, r=17, l=11, d=3, spaces=26...

# Create tensors
input_tensor = torch.LongTensor([7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3, 26, 26, 26, 26, 26, 26, 26, 26, 26])
output_tensor = torch.LongTensor([0, 0, 2, 2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

### Step 5: Ready for Training
- `input_tensor`: Fed into the transformer model
- `output_tensor`: Used as ground truth labels for loss calculation

---

## Key Differences: BEFORE vs BEFOREAFTER

| Aspect | BEFORE | BEFOREAFTER |
|--------|--------|-------------|
| **Information Access** | Only past positions | Entire sequence |
| **Causality** | Causal (autoregressive) | Non-causal (bidirectional) |
| **Difficulty** | Harder (less information) | Easier (more information) |
| **Use Case** | Tests memory/autoregressive ability | Tests full sequence understanding |
| **Example Output** | `[0,0,0,1,0,0,0,1,0,2,0]` | `[0,0,2,2,1,0,0,1,0,2,0]` |

---

## Why This Design Matters

1. **Task Flexibility**: Same codebase handles two different tasks
2. **Model Evaluation**: Tests different capabilities (causal vs bidirectional)
3. **Data Structure**: Consistent format for both tasks
4. **Efficiency**: List comprehension processes all examples at once

---

## Summary

**Lines 90-106** prepare the dataset by:
1. Selecting the task type (BEFORE or BEFOREAFTER)
2. Reading examples from files
3. Generating labels based on the task
4. Converting everything to PyTorch tensors
5. Creating `LetterCountingExample` objects ready for training

The key insight is that **BEFORE** is causal (only past information) while **BEFOREAFTER** is non-causal (full sequence information), making them test different model capabilities.

