# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the official PyTorch implementation of **SAN (Syntax-Aware Network)** for Handwritten Mathematical Expression Recognition, published at CVPR 2022. The system uses a hierarchical attention mechanism with syntax-aware decoding to recognize handwritten mathematical expressions.

---

## Key Commands

### Installation
```bash
pip3 install -r requirement.txt
```

### Data Preparation (CROHME 2016)
```bash
cd data
# Step 1: Prepare image and label files
python3 prepare_crohme_data.py

# Step 2: Generate hybrid tree labels
python3 gen_hybrid_data.py

# Step 3: Convert to pickle format
python3 convert_hybrid_to_pkl.py
```

### Training
```bash
python3 train.py --config config.yaml
```

### Inference
```bash
python3 inference.py --config 14.yaml --image_path data/14_test_images --label_path data/test_caption.txt
```

---

## Architecture Deep Dive

### Core Innovation: Syntax-Aware Hierarchical Decoding

The SAN model's key innovation is a **dual-path decoding strategy** that explicitly models the hierarchical syntactic structure of mathematical expressions:

1. **Forward Path**: Predicts symbols top-down through the expression tree
2. **Backward Path (Child-to-Parent)**: Predicts parent symbols from child+relation pairs as regularization
3. **KL Divergence Loss**: Ensures attention consistency between forward/backward predictions

This approach helps the model understand that mathematical expressions are not flat sequences but have nested hierarchical structures (fractions, subscripts, superscripts, matrices, etc.).

---

## Detailed Architecture Components

### 1. Encoder (`san_model/encoder/densenet.py`)

**DenseNet-based CNN** that extracts visual features from handwritten images:
- Input: Grayscale images (default 320×1600 pixels, configurable)
- Output: Feature maps with 684 channels
- Architecture: 3 dense blocks with transition layers for downsampling
- Key parameters:
  - Growth rate: 24
  - Reduction: 0.5
  - Bottleneck: True
  - Ratio: 16 (downsampling factor)

**Key Classes:**
- `Bottleneck`: Dense block building unit
- `SingleLayer`: Single convolutional layer in dense block
- `Transition`: Downsampling layer between dense blocks
- `DenseNet`: Main encoder combining all components

### 2. Decoder (`san_model/decoder/decoder.py`)

**SAN_decoder** implements syntax-aware hierarchical decoding with three parallel GRU paths:

#### a) Word Prediction Path
Predicts terminal symbols (numbers, letters, operators):
- `word_input_gru` (GRU-α): First GRU layer taking word embedding and parent hidden state
- `word_attention`: Coverage-based attention mechanism
- `word_out_gru` (GRU-β): Second GRU layer producing final hidden state
- `word_convert`: Linear layer mapping to word vocabulary

#### b) Structure Prediction Path
Predicts structural relations (subscript, superscript, fraction, matrix, etc.):
- Shares hidden states with word prediction
- `struct_convert`: Multi-label classifier for active structures
- Uses BCE loss (multiple structures can be active simultaneously)

#### c) Child-to-Parent Path (Training Only)
Reverse prediction for regularization:
- `c2p_input_gru`: Takes concatenated (child embedding + relation embedding)
- `c2p_attention`: Separate attention mechanism for reverse path
- `c2p_out_gru`: Produces hidden state for parent prediction
- `c2p_convert`: Predicts parent symbol

**Key Parameters:**
- Input size: 256
- Hidden size: 256
- Word vocabulary: Variable (depends on dataset)
- Structure types: 5 (subscript, superscript, above, below, inside)
- Dropout: 0.5
- Threshold: 0.5 (for structure prediction during inference)

**Important Constants:**
- `STRUCT_ID`: Token indicating structural relation
- `RIGHT_ID`: Token for horizontal continuation
- `EOS_ID`: End of subtree
- `SUB_ID`, `SUP_ID`: Subscript and superscript structure IDs

### 3. Attention Mechanism (`san_model/decoder/attention.py`)

**Coverage-based attention** that tracks attended image regions:

```python
query = W_h · hidden                                  # Query from decoder state
coverage = W_c · attention_conv(α_sum)                # Coverage from accumulated attention
features = W_e · encoder_features                     # Encoded image features
energy = tanh(query + coverage + features)            # Attention energy
α = softmax(energy)                                   # Attention weights
α_sum = α_sum + α                                     # Accumulate coverage
context = Σ(α · features)                             # Weighted context vector
```

**Key Features:**
- Prevents redundant attention to same regions
- Accumulates attention weights over decoding steps
- Masked by image dimensions to avoid padding regions
- Separate attention for forward and backward paths

**Parameters:**
- Attention dimension: 512
- Channel: 684 (from encoder)
- Hidden: 256 (from decoder)

### 4. Backbone (`san_model/backbone.py`)

Combines encoder and decoder with **four loss components**:

#### Loss Functions:
1. **Word Loss** (CrossEntropy):
   - Predicts symbol at each tree node
   - Applied to `labels[:,:,1]` (child symbol)

2. **Structure Loss** (BCE with Logits):
   - Multi-label prediction of active structures
   - Applied to `labels[:,:,4:]` (structure vector)
   - Masked by `labels_mask` if provided

3. **Parent Loss** (CrossEntropy, training only):
   - Child-to-parent prediction loss
   - Applied to `labels[:,:,3]` (parent symbol)

4. **KL Loss** (KL Divergence, training only):
   - Measures consistency between forward and backward attention maps
   - Formula: `KL(α_child || α_parent)`
   - Ensures child attention is consistent with parent's attention
   - Shifted by one timestep (child uses parent's previous attention)

**Forward Pass:**
```python
# Encoder
cnn_features = encoder(images)

# Decoder
word_probs, struct_probs, word_alphas, struct_alphas, c2p_probs, c2p_alphas =
    decoder(cnn_features, labels, images_mask, labels_mask, is_train=True)

# Losses
word_loss = CrossEntropy(word_probs, labels[:,:,1])
struct_loss = BCE(struct_probs, labels[:,:,4:])
parent_loss = CrossEntropy(c2p_probs, labels[:,:,3])
kl_loss = KL(child_alphas, parent_alphas)

total_loss = word_loss + struct_loss + parent_loss + kl_loss
```

---

## Data Representation: Hybrid Tree

The model uses a **hybrid tree** representation where mathematical expressions are linearized into a sequence of tree nodes. Each node contains:

### Label Tensor Structure:
- `labels[:,:,0]`: Line ID (unused during training, for reference)
- `labels[:,:,1]`: Child symbol ID (terminal to predict)
- `labels[:,:,2]`: Parent line ID (for tree traversal)
- `labels[:,:,3]`: Parent symbol ID (relation type: right, sub, sup, etc.)
- `labels[:,:,4:]`: Multi-hot structure vector (5D: which relations are active)

### Example: Expression `2x_0`

```
Line 0: word='2',      parent=-1, relation='START',  structs=[]
Line 1: word='x',      parent=0,  relation='right',  structs=[]
Line 2: word='STRUCT', parent=1,  relation='x',      structs=[subscript]
Line 3: word='0',      parent=2,  relation='sub',    structs=[]
```

### Example: Expression `\frac{a}{b}`

```
Line 0: word='frac',   parent=-1, relation='START',  structs=[]
Line 1: word='STRUCT', parent=0,  relation='frac',   structs=[above, below]
Line 2: word='a',      parent=1,  relation='above',  structs=[]
Line 3: word='b',      parent=1,  relation='below',  structs=[]
```

### Tree Traversal During Training:

The decoder maintains a **parent hidden states history** indexed by line ID:
- `parent_hiddens[0]`: Initial state (from encoder features)
- `parent_hiddens[i+1]`: Hidden state after processing line i
- At each step, retrieves parent hidden state using `labels[:,i,2]` (parent line ID)
- Teacher forcing: uses ground truth parent symbols and relations

---

## Training Process

### Configuration (`training/config/config.yaml`)

**Optimizer Settings:**
- Optimizer: Adadelta
- Learning rate: 1.0
- LR decay: Cosine schedule
- Epsilon: 1e-6
- Weight decay: 1e-4
- Gradient clipping: 100

**Training Hyperparameters:**
- Batch size: 8
- Epochs: 200
- Workers: 0 (data loading)
- Dropout: 0.5
- Image size: 320×1600 (height × width)
- Image channel: 1 (grayscale)

**Model Architecture:**
- Encoder: DenseNet
  - Output channels: 684
  - Growth rate: 24
  - Reduction: 0.5
- Decoder: SAN_decoder
  - Cell: GRU
  - Input size: 256
  - Hidden size: 256
  - Attention dim: 512

**Data Paths:**
- Train images: `data/train_image.pkl`
- Train labels: `data/train_label.pkl`
- Eval images: `data/test_image.pkl`
- Eval labels: `data/test_label.pkl`
- Vocabulary: `data/word.txt`

### Training Loop (`training/training.py`)

**Per Epoch:**
1. **Forward Pass**:
   ```python
   probs, loss = model(images, image_masks, labels, label_masks, is_train=True)
   word_probs, struct_probs = probs
   word_loss, struct_loss, parent_loss, kl_loss = loss
   total_loss = word_loss + struct_loss + parent_loss + kl_loss
   ```

2. **Backward Pass**:
   - Uses gradient accumulation (can be configured)
   - Uses automatic mixed precision (AMP) with GradScaler
   - Gradient clipping to prevent explosion

3. **Metrics Tracking**:
   - **Word accuracy**: % of correctly predicted symbols
   - **Structure accuracy**: % of correctly predicted structure relations
   - **Expression rate**: % of perfectly recognized expressions
   - Tracked separately for train and eval sets

4. **Checkpointing**:
   - Saves best model based on eval loss
   - Also saves optimizer state if configured
   - Path: `checkpoints/{experiment_name}/best.pth`

5. **TensorBoard Logging**:
   - Losses (word, struct, parent, KL)
   - Accuracies (word, struct, expression)
   - Learning rate
   - Saved to `logs/` directory

### Dataset (`training/dataset.py`)

**HYBTr_Dataset** class:
- Loads pre-processed pickle files containing images and labels
- **Image preprocessing**:
  - Convert to grayscale
  - Resize to target dimensions (with aspect ratio preservation)
  - Pad to fixed width (max_width)
  - Normalize to [0, 1]
- **Label preprocessing**:
  - Already in hybrid tree format from data preparation scripts
  - Convert word strings to vocabulary indices
- **Collate function**:
  - Dynamic padding to longest sequence in batch
  - Creates image masks (for variable width images)
  - Creates label masks (for variable length labels)
  - Returns tensors ready for GPU

**get_dataset** function:
- Creates train and eval dataloaders
- Configures batch size, workers, shuffling
- Handles vocabulary loading from `word.txt`

---

## Inference Mode

During inference (`is_train=False`), the model uses **stack-based beam search**:

### Algorithm:

1. **Initialize**:
   - Start with `<START>` token embedding
   - Initial hidden state from encoder features
   - Empty structure stack: `struct_list = []`

2. **For each decoding step**:

   a) **Predict word**:
   ```python
   hidden = word_input_gru(embedding, parent_hidden)
   context, alpha, alpha_sum = attention(features, hidden, alpha_sum, mask)
   hidden = word_out_gru(context, hidden)
   word_prob = word_convert(hidden)
   word = argmax(word_prob)
   ```

   b) **Handle different word types**:

   - **If word == STRUCT**:
     - Predict structures: `structs = sigmoid(struct_convert(hidden))`
     - Filter by threshold (default 0.5)
     - Push active structures to stack (in reverse order)
     - Format: `(relation_id, hidden_state, alpha_sum)`
     - Pop first structure to continue

   - **If word == EOS** (end of subtree):
     - If stack is empty: terminate decoding
     - Else: pop structure from stack to continue parent subtree

   - **Else** (regular symbol):
     - Use word as next embedding
     - Update parent_hidden = hidden (for next child)

3. **Terminate** when:
   - Predict EOS and stack is empty
   - Reach maximum steps (configurable)

### Key Differences from Training:

- **No teacher forcing**: Uses predicted symbols, not ground truth
- **No backward path**: Child-to-parent prediction not used
- **No KL loss**: Only forward attention
- **Greedy decoding**: Takes argmax (can be extended to beam search)
- **Stack-based**: Manages incomplete structures dynamically

---

## Key Files Structure

```
san_model/
├── backbone.py              # Main model combining encoder+decoder with loss functions
├── encoder/
│   ├── __init__.py
│   └── densenet.py          # DenseNet CNN encoder (3 dense blocks)
└── decoder/
    ├── __init__.py
    ├── decoder.py           # SAN decoder with hierarchical attention and dual-path GRU
    └── attention.py         # Coverage-based attention mechanism

training/
├── __init__.py
├── train.py                 # Main training script (entry point)
├── training.py              # Train/eval functions with metrics tracking
├── dataset.py               # HYBTr_Dataset class and dataloader creation
└── config/
    ├── config.yaml          # Main configuration file
    ├── config_mathwriting.yaml
    ├── chrome2013.yaml
    └── 14.yaml

data_tools/                  # Data preparation scripts (not in san_model/)
inference/                   # Inference scripts and utilities
checkpoints/                 # Saved model weights
logs/                        # TensorBoard logs
```

---

## Distinguishing Features

1. **Syntax-Aware Decoding**:
   - Explicitly models hierarchical structure of math expressions
   - Uses parent-child relationships to guide attention

2. **Dual-Path Regularization**:
   - Forward prediction (parent → child)
   - Backward prediction (child → parent)
   - KL divergence ensures consistency between attention maps

3. **Coverage-Based Attention**:
   - Accumulates attention weights over timesteps
   - Prevents redundant attention to same image regions
   - Separate attention for each path

4. **Multi-Task Learning**:
   - Jointly predicts symbols (word_probs) and structures (struct_probs)
   - BCE loss for multi-label structure prediction

5. **Teacher Forcing with Tree Structure**:
   - During training, uses ground truth to retrieve parent hidden states
   - Maintains history of all parent states indexed by line ID
   - Enables efficient parallel processing of entire sequence

6. **Stack-Based Inference**:
   - Dynamically manages incomplete structural relations
   - Supports arbitrary nesting depth
   - No pre-defined maximum structure complexity

7. **Hierarchical GRU Architecture**:
   - Two-level GRU for both forward and backward paths
   - First level (input_gru) incorporates parent/partner information
   - Second level (out_gru) incorporates attention context

---

## Important Notes

- The model requires CUDA-compatible GPU (originally uses PyTorch 1.6.0+cu101)
- For Python 3.11+, use `requirements_python311.txt` for compatible package versions
- On macOS with Apple Silicon, PyTorch will use MPS (Metal Performance Shaders) instead of CUDA
- Pre-trained checkpoint available at `checkpoints/SAN_decoder/best.pth`
- Supports CROHME and HME100K datasets
- No explicit test suite - evaluation is done during training/inference with metrics:
  - Word-level accuracy
  - Structure-level accuracy
  - Expression-level recognition rate (ExpRate)

## Performance Expectations

Based on the CVPR 2022 paper:
- **CROHME 2014**: ~59% expression recognition rate
- **CROHME 2016**: ~58% expression recognition rate
- **CROHME 2019**: ~60% expression recognition rate
- Competitive with or better than previous state-of-the-art methods
- Especially strong on expressions with complex nested structures

---

## Synthetic Data Generation

For documentation on the synthetic handwritten math expression generation system, see:
- **`data_tools/dataset_prep/synthetic/CLAUDE.md`** - Complete pipeline documentation

The system generates training data by compiling LaTeX to DVI format, extracting glyph positions, and replacing symbols with real handwritten strokes from a library of 6,423 InkML files.
