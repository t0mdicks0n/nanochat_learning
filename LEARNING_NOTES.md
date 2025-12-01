# Learning Notes: How LLMs Work

These are my notes from exploring the nanochat repository to understand how Large Language Models work from the ground up. The goal was to train a small model locally and understand each step of the pipeline.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Tokenization](#tokenization)
3. [Pre-training (Base Training)](#pre-training-base-training)
4. [The Transformer Architecture](#the-transformer-architecture)
5. [Attention Mechanism](#attention-mechanism)
6. [Midtraining and Fine-tuning](#midtraining-and-fine-tuning)
7. [Inference](#inference)
8. [Advanced Topics](#advanced-topics)
9. [Key Takeaways](#key-takeaways)

---

## The Big Picture

An LLM is trained in stages:

```
Raw Text → Tokenizer Training → Pre-training → Midtraining → Fine-tuning → Chat Model
```

Each stage uses the same core technique: **next-token prediction**. The model learns to predict what comes next in a sequence, and through billions of predictions, it learns language, facts, and reasoning patterns.

---

## Tokenization

**Goal:** Convert text to integers so neural networks can process it.

### The Naive Approaches (and why they fail)

**Approach 1: One token per character**

```
"the cat" → [t, h, e, " ", c, a, t] → 7 tokens
```

Problems:
- Sequences become very long
- A 1024-token context window only fits ~1024 characters
- Inefficient - common words like "the" take 3 tokens every time

**Approach 2: One token per word**

```
"the cat" → [the, cat] → 2 tokens
```

Problems:
- Vocabulary explodes (millions of unique words)
- Unknown words break the system ("cryptocurrency" wasn't in training data)
- What about typos like "teh"?

**The comparison:**

| Approach | Vocabulary Size | Handles Any Text? | Efficient? |
|----------|-----------------|-------------------|------------|
| Per-character | 256 | Yes | No - sequences too long |
| Per-word | Infinite | No - unknown words fail | No - huge vocabulary |
| Byte Pair Encoding | 65,536 (fixed) | Yes - falls back to bytes | Yes - short sequences |

### Byte Pair Encoding (the actual solution)

Instead of one token per word, we use **subword tokenization**. The algorithm:

1. Start with 256 tokens (all possible bytes)
2. Scan training data and count which byte pairs appear most often
3. Merge the most frequent pair into a new token
4. Repeat until you have ~65,000 tokens

**What are "256 bytes"?**

A byte is 8 bits, which can represent 256 different values (0-255):

```
0-127:   ASCII characters
         0-31:    Control characters (newline, tab, etc.)
         32-126:  Printable ASCII (a-z, A-Z, 0-9, punctuation, space)
         127:     Delete character

128-255: Extended bytes (used for UTF-8 encoded characters like é, 中, emoji)
```

Starting with all 256 bytes guarantees the tokenizer can handle **any text**:
- "Hello" → ASCII bytes
- "café" → "caf" + 2 bytes for "é" (UTF-8)
- "🔥" → 4 bytes (UTF-8 emoji)

Even if Byte Pair Encoding never merged certain characters, it can always fall back to individual bytes.

**The merge process builds up from bytes:**

```
Iteration 1:    "t" + "h" → "th"           (token 256)
Iteration 2:    "th" + "e" → "the"         (token 257)
Iteration 3:    " " + "the" → " the"       (token 258)
...
Iteration 65280: final merge               (token 65535)
```

Final vocabulary: 256 bytes + 65,280 learned merges = 65,536 tokens.

**Only frequent patterns get their own token.** With 65,536 slots and millions of unique words, rare words don't make the cut:

```python
# What the tokenizer produces:
"the"            → 1 token   (common word, merged early)
"President"      → 1 token   (common in news/educational text)
"Bitcoin"        → 1 token   (appears enough in training data)
"cryptocurrency" → 3 tokens  ["c", "rypt", "ocurrency"] (rare, stays split)
"TomDickson"     → 3 tokens  ["Tom", "D", "ickson"] (never seen, uses known pieces)
```

The tokenizer doesn't "know" words - it knows **byte patterns** that were frequent in training.

**Code reference:** The merge algorithm is in [rustbpe/src/lib.rs](rustbpe/src/lib.rs) (lines 164-256).

**Example result:**

```python
"the"            → 1 token   (common word, merged early)
"President"      → 1 token   (common in training data)
"cryptocurrency" → 3 tokens  ["c", "rypt", "ocurrency"] (rare, stays split)
"TomDickson"     → 3 tokens  ["Tom", "D", "ickson"] (never seen, uses known pieces)
```

**Key insight:** The tokenizer is a fixed lookup table after training. It doesn't learn or add new entries when encoding text - it just breaks unknown words into known pieces.

**Why this matters:**
- Fixed vocabulary (65,536 tokens) can encode ANY text
- Common words are single tokens (efficient)
- Falls back to individual bytes for unknown characters (works for any language, emoji, etc.)

**Code reference:** [nanochat/tokenizer.py](nanochat/tokenizer.py) contains the Python wrapper around the tokenizer.

---

## Pre-training (Base Training)

**Goal:** Teach the model language patterns and knowledge by predicting the next token.

### The Training Data: FineWeb-Edu

The pre-training data comes from **FineWeb-Edu**, a high-quality dataset of educational web content. It's derived from CommonCrawl (a massive web scrape) but filtered to keep only educational content - articles, tutorials, explanations, academic text.

**Exploring the data:**

```python
import pandas as pd
from pathlib import Path

data_dir = Path.home() / '.cache/nanochat/base_data'
shards = list(data_dir.glob('*.parquet'))
df = pd.read_parquet(shards[0])

print(f"Documents in one shard: {len(df)}")  # ~53,000
print(f"Columns: {df.columns.tolist()}")     # ['text']
```

**Example document from the training data:**

```
NEW YORK (AP) — Former President Donald Trump showed up Wednesday for
questioning under oath in New York's civil investigation into his business
practices. But he quickly made clear he wouldn't be answering.

The ex-president issued a statement saying he had done nothing wrong but was
invoking the Fifth Amendment's protection against self-incrimination. It's a
constitutional right that gets high-profile exposure in settings from Congress
to TV crime shows, but there are nuances. Here's what it means — and doesn't
— to "plead (or 'take') the Fifth."

WHAT IS 'THE FIFTH'?
The Fifth Amendment to the U.S. Constitution establishes a number of rights
related to legal proceedings, including that no one "shall be compelled in any
criminal case to be a witness against himself."
```

This is the kind of text the model learns from - factual, educational, well-written English.

**Scale:**
- Full dataset: ~100 billion tokens
- My training run: Downloaded 4 shards (~200 million tokens)
- Each shard: ~53,000 documents, ~50 million tokens

### Self-Supervised Learning

The clever trick: raw text labels itself. Given a sequence:

```
"NEW YORK (AP) — Former President"
```

We automatically create training pairs:

| Input (context) | Target (next token) |
|-----------------|---------------------|
| NEW | YORK |
| NEW YORK | ( |
| NEW YORK ( | AP |
| NEW YORK (AP | ) |
| NEW YORK (AP) — | Former |
| NEW YORK (AP) — Former | President |

No human labeling needed. Every piece of text becomes millions of training examples.

### The Training Loop

```
1. Grab 1024 tokens from training data
2. Model predicts next token at each position
3. Calculate loss (how wrong was it?)
4. Backpropagation: figure out which weights caused errors
5. Update weights slightly to reduce error
6. Repeat millions of times
```

**Code reference:** [scripts/base_train.py](scripts/base_train.py) contains the training loop.

### Understanding Loss

Loss measures how wrong the model's predictions were.

The model outputs probabilities for all 65,536 tokens. If it said the correct token has 2% probability but we wanted 90%, the loss is high.

**Starting loss:** With random weights, the model guesses randomly. The loss starts at:
```
-log(1/65536) ≈ 11.09
```
This is the loss of random guessing over 65,536 tokens.

**Training progress:**
| Steps | Loss | Meaning |
|-------|------|---------|
| 0 | 11.09 | Random guessing |
| 500 | 6.82 | Better than random, still bad |
| Millions | ~3-4 | Good predictions |

### Why Scale Matters

My training run:
- 500 steps, 4 layers, 37M parameters
- Result: Complete nonsense

Karpathy's hosted model:
- 33 hours on 8x H100 GPUs, 32 layers, 1.9B parameters, 38B tokens
- Result: Usable (but still "like talking to a child")

GPT-4: Likely $50-100 million to train.

---

## The Transformer Architecture

The transformer is implemented from scratch in [nanochat/gpt.py](nanochat/gpt.py) - only ~300 lines.

### High-Level Flow

```
Input tokens: [35064, 51897, 372]
        ↓
┌─────────────────────────────────┐
│  Embedding                      │  Token IDs → Vectors (256 dimensions)
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  Block 1                        │ ─┐
│    - Attention                  │  │
│    - Feed-forward network       │  │  Repeat n_layer times
└─────────────────────────────────┘  │  (4 for my model, 32 for real ones)
        ↓                            │
│  Block 2, 3, 4...               │ ─┘
        ↓
┌─────────────────────────────────┐
│  Output head                    │  Vectors → 65,536 probabilities
└─────────────────────────────────┘
        ↓
Output: probability for each possible next token
```

### Step 1: Embedding (lines 143, 256)

Each token ID becomes a vector of numbers:

```python
x = self.transformer.wte(idx)  # Token ID → 256-dimensional vector
```

```
Token 35064 ("NEW")  → [0.23, -0.45, 0.12, ..., 0.87]  (256 numbers)
Token 51897 ("YORK") → [0.11, 0.33, -0.28, ..., 0.54]  (256 numbers)
```

These numbers represent the "meaning" of each token. The model learns these during training.

### Step 2: Transformer Blocks (lines 126-135)

Each block does two things:

```python
def forward(self, x, cos_sin, kv_cache):
    x = x + self.attn(norm(x), cos_sin, kv_cache)  # Attention
    x = x + self.mlp(norm(x))                       # Feed-forward
    return x
```

**Attention:** "Which previous tokens should I pay attention to?"

**Feed-forward network:** A simple neural network that processes what attention found. It expands dimensions (256 → 1024), applies a non-linearity, then compresses back (1024 → 256). This gives the network "room to think."

```python
class MLP(nn.Module):
    def forward(self, x):
        x = self.c_fc(x)       # Expand: 256 → 1024
        x = F.relu(x).square() # Non-linearity
        x = self.c_proj(x)     # Compress: 1024 → 256
        return x
```

### Step 3: Output Head (lines 267-270)

Convert the final 256-dimensional vector to 65,536 scores:

```python
logits = self.lm_head(x)  # 256 dims → 65,536 scores
```

Higher score = model thinks that token is more likely next.

---

## Attention Mechanism

**Code reference:** [nanochat/gpt.py](nanochat/gpt.py) lines 51-110.

Attention answers: "When predicting the next token, how much should I care about each previous token?"

### Example

```
"The cat sat on the ___"
```

When predicting the blank:
- "cat" → very relevant (it's the subject)
- "on" → very relevant (implies location coming)
- "The" → less relevant (just grammar)

### How It Works: Query, Key, Value

Each token gets transformed into three vectors:

```python
q = self.c_q(x)  # Query: "What am I looking for?"
k = self.c_k(x)  # Key: "What do I contain?"
v = self.c_v(x)  # Value: "What information should I pass along?"
```

**Important:** These are just vectors of numbers with no human-interpretable meaning. The model learns (through training) what to put in each one.

### Step-by-Step

**1. Create Query, Key, Value for each token:**

```
Token "cat" embedding: [0.23, -0.45, 0.12, ...]

After learned projections:
  Q_cat = [0.82, -0.31, ...]   ← 128 numbers
  K_cat = [0.14, 0.67, ...]    ← 128 numbers
  V_cat = [0.91, 0.03, ...]    ← 128 numbers
```

**2. Compare Queries to Keys (dot product):**

For the token "on", compare its Query to all previous tokens' Keys:

```
Q_on · K_the = 0.1  (low = not relevant)
Q_on · K_cat = 0.7  (high = relevant!)
Q_on · K_sat = 0.5  (medium)
Q_on · K_on  = 0.3  (some relevance)
```

After softmax, these become weights: `[0.05, 0.45, 0.30, 0.20]`

**3. Weighted sum of Values:**

```
Output_on = 0.05 × V_the + 0.45 × V_cat + 0.30 × V_sat + 0.20 × V_on
```

The output for "on" now contains information from all relevant previous tokens, especially "cat".

### Causal Attention

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

`is_causal=True` means tokens can only look backwards, not forwards. You can't cheat by looking at the answer!

---

## Midtraining and Fine-tuning

**Goal:** Same next-token prediction, but on conversation-formatted data.

### The Difference from Pre-training

**Pre-training data:** Raw text from the web
```
NEW YORK (AP) — Former President Donald Trump showed up Wednesday...
```

**Midtraining data:** Conversation-formatted text with special tokens
```
<|user_start|>What is the capital of France?<|user_end|>
<|assistant_start|>The capital of France is Paris.<|assistant_end|>
```

### Midtraining Data Sources

**Code reference:** [scripts/mid_train.py](scripts/mid_train.py) lines 98-106.

```python
train_dataset = TaskMixture([
    SmolTalk(split="train"),           # 460K general conversations
    MMLU(subset="auxiliary_train"),    # 100K multiple choice questions
    GSM8K(subset="main", split="train"), # 8K math problems
    CustomJSON(filepath=identity_conversations_filepath),  # 1K identity Q&A
    SimpleSpelling(size=200000),       # 200K spelling tasks
    SpellingBee(size=80000),           # 80K "count letters" tasks
])
```

| Dataset | Size | What It Contains |
|---------|------|------------------|
| SmolTalk | 460K | General conversations (from HuggingFace) |
| MMLU | 100K | Multiple choice Q&A across 57 subjects |
| GSM8K | 8K | Grade school math with step-by-step solutions |
| CustomJSON | 1K | Synthetic "who are you?" identity conversations |
| SimpleSpelling | 200K | "Spell the word 'apple'" |
| SpellingBee | 80K | "How many 'r' in 'strawberry'?" |

**Where does this data come from?**
- **SmolTalk:** Curated by HuggingFace from various sources
- **MMLU/GSM8K:** Academic benchmarks created by researchers
- **CustomJSON:** You can write your own! (see `dev/gen_synthetic_data.py`)
- **Spelling tasks:** Auto-generated programmatically

### What the Model Learns

From this data, the model learns:
- "After `<|user_start|>` comes a question"
- "After `<|user_end|>` comes `<|assistant_start|>`"
- "I should answer, then output `<|assistant_end|>`"

### This Is Why ChatGPT Can Chat

The chat interface just wraps your input with special tokens:

```
1. You type: "What is 2+2?"
2. System wraps it: "<|user_start|>What is 2+2?<|user_end|><|assistant_start|>"
3. Model predicts: "4<|assistant_end|>"
4. System shows you: "4"
```

Same next-token prediction, just trained on conversation format.

---

## Inference

**Code reference:** [nanochat/gpt.py](nanochat/gpt.py) lines 244-276 (forward pass) and 278-307 (generation).

### Training vs Inference

The same `forward()` method is used, but differently:

**Training:**
```python
loss = model(tokens, targets=next_tokens)  # Returns loss
loss.backward()  # Update weights
```

**Inference:**
```python
logits = model(tokens)  # Returns 65,536 scores
next_token = pick_highest(logits)  # Choose one
```

### Autoregressive Generation

```python
def generate(self, tokens, max_tokens, ...):
    for _ in range(max_tokens):
        logits = self.forward(ids)          # Predict next
        next_token = pick_token(logits)     # Sample one
        ids = torch.cat([ids, next_token])  # Append
        yield next_token                    # Output
```

Generate one token, append it, generate the next, repeat.

---

## Advanced Topics

### The Generality of Transformers

The transformer architecture doesn't care what the tokens represent. It just learns patterns in sequences. This means the same architecture works for:

```
Text:   [the, cat, sat, ???]        → predict next word
Chess:  [e4, e5, Nf3, ???]          → predict next move
Music:  [C4, E4, G4, ???]           → predict next note
Code:   [def, foo, (, ???]          → predict next token
```

**Chess example:** Researchers trained transformers on millions of chess games written in standard notation (e.g., "1. e4 e5 2. Nf3 Nc6..."). The model learns to predict the next move - same next-token prediction, just on chess moves instead of words. Surprisingly, the model develops an internal "understanding" of the board state, even though it only sees move text.

**Image generation:** To generate images with transformers:
1. Convert images to tokens (split into patches, or use a learned discrete tokenizer)
2. Train on paired data: "a fluffy cat" → [image tokens]
3. Same next-token prediction

The text-image pairing is essential - without it, you can generate images but can't control *what* they show.

**The insight:** Next-token prediction is a general learning algorithm. Anything that can be represented as a sequence can potentially be learned by a transformer.

### Reasoning Models (like o1, Claude with extended thinking)

The idea: let the model "think" before answering by generating intermediate tokens.

**Without reasoning:**
```
Q: What is 17 × 24?
A: 408 (immediate guess, might be wrong)
```

**With reasoning:**
```
Q: What is 17 × 24?
Thinking: 17 × 20 = 340, 17 × 4 = 68, 340 + 68 = 408
A: 408
```

More tokens = more "compute time" on the problem. The model learns to output thinking steps through training on data that shows reasoning.

### Retrieval Augmented Generation vs Fine-tuning

**Fine-tuning:** Bake knowledge into model weights
- Expensive, can cause "catastrophic forgetting"
- Hard to update

**Retrieval Augmented Generation:** Paste relevant documents into the prompt at inference time
- Model doesn't need to memorize - just reads and reasons
- Easy to update (just update the document database)
- Most companies use this approach

### Mixture of Experts

Instead of one big network, have multiple specialized smaller networks:

```
Token → Router → Expert 1 (maybe good at code)
              → Expert 2 (maybe good at math)
              → Expert 3 (maybe good at language)
```

The router learns which expert to use. You get a huge model but only run part of it per token.

### Image Generation

Transformers work on sequences. To generate images:

1. Convert images to tokens (patches or learned discrete tokens)
2. Use the same transformer architecture
3. Train on paired data: "a fluffy cat" → [image tokens]

The text-image pairing is essential - without it, you can't control what the model generates.

**How do you get text-image pairs at scale?** Web scraping. People crawl billions of images from the web and extract the `alt` tags from the HTML (e.g., `<img src="cat.jpg" alt="a fluffy orange cat sitting on a couch">`). The alt text becomes the training label. This is how datasets like LAION-5B were built - 5 billion image-text pairs scraped from the web. It's noisy (not every alt tag is accurate), but at scale, the model learns the patterns anyway.

---

## Key Takeaways

1. **Tokenization is compression.** Byte Pair Encoding creates a fixed vocabulary that efficiently represents any text by breaking rare words into common pieces.

2. **Pre-training is self-supervised.** The data labels itself - predict the next token. No human labeling needed for billions of training examples.

3. **The transformer is simple.** Embedding → Attention (which tokens matter?) → Feed-forward (process what we found) → Output probabilities. That's it.

4. **Attention is learned relevance.** Query/Key/Value are just number vectors. The model learns through training what makes tokens relevant to each other.

5. **Fine-tuning is the same algorithm, different data.** Conversation format is just special tokens. The model learns the pattern through next-token prediction.

6. **Scale is everything.** My 2-minute training produced nonsense. Real models need millions of steps, billions of parameters, and significant compute.

---

## Running It Locally (on a MacBook)

I ran the full pipeline on my MacBook to understand each step. The model is unusable, but the process is identical to how real LLMs are built - just at 1/1,000,000th the scale.

### Environment Setup

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Rust (for tokenizer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Create virtual environment
uv venv
source .venv/bin/activate
uv sync --extra cpu

# Build the Rust tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Set environment variables
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
```

### Step 1: Train the Tokenizer

```bash
# Download training data (4 shards)
python -m nanochat.dataset -n 4

# Train Byte Pair Encoding tokenizer
python -m scripts.tok_train --max_chars=1000000000

# Evaluate compression ratio
python -m scripts.tok_eval
```

### Step 2: Pre-training (Base Training)

```bash
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --eval_every=100 \
    --eval_tokens=4096 \
    --core_metric_every=-1 \
    --sample_every=100 \
    --num_iterations=500
```

**Parameters explained:**
- `--depth=4`: 4 transformer layers (tiny model, ~37M parameters)
- `--max_seq_len=1024`: Context window of 1024 tokens
- `--device_batch_size=1`: Process 1 sequence at a time (Mac memory limit)
- `--num_iterations=500`: 500 optimization steps
- `--core_metric_every=-1`: Disable memory-hungry benchmarks

### Step 3: Midtraining

```bash
python -m scripts.mid_train \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --num_iterations=100
```

### Step 4: Chat with the Model

```bash
python -m scripts.chat_cli -i mid
```

---

## Results: What I Got

### Training Progress

| Stage | Starting Loss | Ending Loss | Time |
|-------|---------------|-------------|------|
| Pre-training (500 steps) | 11.09 (random) | 6.82 | ~1 min |
| Midtraining (100 steps) | ~6.5 | 5.20 | ~15 sec |

The loss dropped from 11.09 (random guessing over 65,536 tokens) to 6.82, showing the model learned *something*.

### Sample Outputs During Pre-training

**At step 50** (just started learning):
```
<|bos|>The capital of France is also be aisbe biminiensis can be aisbe biminiensis
<|bos|>The chemical symbol of gold is aisbe biminiensis can be aisbe biminiensis,
```
Complete nonsense - "aisbe biminiensis" is a species of copepod it saw in training data!

**At step 500** (slightly better):
```
<|bos|>The capital of France is a number of the world, the world, the world, the world, the
<|bos|>The planets of the solar system are: the world, the world, the world, the world, the
<|bos|>If 5*x + 3 = 13, then x is a new, 2020202020200. The 200
```
It learned some sentence structure and repetition patterns, but still nonsense.

### Chat Output After Midtraining

**Input:** "Hi"

**Output:**
```
This function uses a two variables is a value of the function `nict`):
                        return 2,2
      return 4:
      if n) = 1.(b)^2) = 3.5) = 5
```python
def count_a = 112)
     return (3) = 2 - 50
```

The model learned the conversation format (it tries to respond) but mixes up code, math, and conversation because it barely trained.

**Input:** "What do you know about PSQL?"

**Output:**
```
You are you on the task is the response to the word 'n:s':
                      if you are the answer.<|user_end|><|assistant_start|>To find the answer
to the word with you, I may be a dictionary in more in the idea of the task.
```

It correctly uses the special tokens (`<|user_end|>`, `<|assistant_start|>`) showing it learned the format. But the content is gibberish.

### Why It's So Bad

| Resource | My Training | Karpathy's $800 Model | Real LLMs |
|----------|-------------|----------------------|-----------|
| Parameters | 37M | 1.9B | 70B-1T |
| Tokens trained | ~500K | 38B | 1-15T |
| Layers | 4 | 32 | 80-120 |
| Training time | 2 minutes | 33 hours | Months |
| Cost | $0 | ~$800 | $50-100M |

The model learned the *structure* (special tokens, sentence patterns) but has no real knowledge because it barely saw any data.

### The Insight

Even though my model is useless, **the process is identical to GPT-4**:
1. Tokenize text with Byte Pair Encoding
2. Pre-train on next-token prediction
3. Fine-tune on conversation format
4. Chat via autoregressive generation

Scale is the only difference.

What's really interesting is that Karpathy's hosted model feels surprisingly good to converse with. I only played with it for a couple of minutes and asked it basic questions, but if someone gave me that before I had seen ChatGPT then I would have been very impressed. You could probably solve real world problems such as generating SEO slop articles, and that's for a model which cost $800 to train yourself and that you have complete control over.

One thing which is kind of wild is that Karpathy's simple $800 model outperforms GPT-2 from OpenAI in 2019. It is rumoured that OpenAI spent $50,000-$100,000+ in 2019 on compute for training GPT-2. We get the same capability in 2025 for $800!

This is not primarily from algorithmic improvements. The transformer architecture is essentially unchanged from 2017. Scale and efficiency are what improved:

| GPU | Year | Performance | Cloud Cost |
|-----|------|-------------|------------|
| V100 (GPT-2 era) | 2017 | 125 TFLOPS | ~$3/hr |
| H100 (now) | 2022 | 1,979 TFLOPS | ~$3/hr |

H100s are ~15x faster at the same price. What took days now takes hours.

**Why $800 beats $50,000+ from 2019:**
- Hardware got 10-15x faster at the same price
- Software got more efficient (Flash Attention, better compilers)
- Knowledge of what works (data quality matters more than we knew)

---

## Resources for Further Learning

- [Andrej Karpathy's "Neural Networks: Zero to Hero"](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) - Builds up to this exact code
- [nanochat repository](https://github.com/karpathy/nanochat) - The code these notes reference
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper

---

*Notes from exploring nanochat, December 2025*
