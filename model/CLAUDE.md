# Deep Learning Model 

## Key Commands

### Training
```bash
cd training
python3 train.py --config config.yaml
```

### Inference
```bash
python3 inference.py --config config.yaml --image_path ../../data/test_images --label_path ../../data/test_caption.txt
```

---

## Architecture Deep Dive

### Core Innovation: Syntax-Aware Hierarchical Decoding

The model explicitly models the hierarchical syntactic structure of mathematical expressions. It predicts symbols top-down through the expression tree. This approach helps the model understand that mathematical expressions are not flat sequences but have nested hierarchical structures (fractions, subscripts, superscripts, matrices, etc.).

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

**Key Parameters:**
- Input size: 256
- Hidden size: 256
- Word vocabulary: Variable (depends on dataset)
- Structure types: 7 (subscript, superscript, above, below, inside, left-superscript, right)
- Dropout: 0.5
- Threshold: 0.5 (for structure prediction during inference)

**Important Constants:**
- `STRUCT_ID`: Token indicating structural relation
- `RIGHT_ID`: Token for horizontal continuation
- `EOS_ID`: End of subtree
- `SUB_ID`, `SUP_ID`: Subscript and superscript structure IDs

### 3. Attention Mechanism (`san_model/decoder/attention.py`)

**Coverage-based attention** that tracks attended image regions:
It uses two coverage accumulation vectors:
- a_sum_parent: the sum of all active parents. An active parent is the previously generated symbol and the construct the parser is currently working on. E.g. If inside a denominator of a fraction, the \frac and corresponding 'struct' elements are active parents.
- a_sum_completed: the sum of all symbols and constructs that are dealt with completely.
The idea is that an a_sum_parent should lead to boosted attention, while a_sum_completed should penalise attention.

```python
query = W_h · hidden                                                       # Query from decoder state
coverage_active = W_act · attention_conv(α_sum_active)                     # Coverage from accumulated active attention
coverage_completed = W_com · attention_conv(α_sum_completed)               # Coverage from accumulated active attention
features = W_e · encoder_features                                          # Encoded image features
energy = tanh(query + features + coverage_active + coverage_completed)     # Attention energy
α = softmax(energy)                                                        # Attention weights
context = Σ(α · features)                                                  # Weighted context vector
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

3. **EOS loss** 
the End Of Sequence symbol has a global influence on predication as it can end subsequent processing. Therefore, we should use an amplified loss for this symbol w.r.t. other symbols.

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
eos_loss = ...

total_loss = word_loss + struct_loss + eos_loss
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
- Learning rate: 0.3
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

**Data Loader:**
Creates buckets of math images with comparable sizing. This limits the amount of padding and allows using large minibatches for short expressions and small minibatches for large expressions. This improves speed of training with a factor of 3.

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
- **Greedy decoding**: Takes argmax (can be extended to beam search)
- **Stack-based**: Manages incomplete structures dynamically

---

## Key Files Structure

```
model/                       # Core ML model package (inference + training)
├── config.yaml              # Main configuration for inference
├── inference.py             # Standalone CLI inference tool
├── inference/               # Inference-specific modules
│   ├── __init__.py
│   ├── Backbone.py          # Inference-optimized backbone
│   └── san_decoder.py       # Inference-optimized decoder
├── san_model/               # Core model architecture
│   ├── __init__.py
│   ├── backbone.py          # Main model combining encoder+decoder with loss functions
│   ├── encoder/
│   │   ├── __init__.py
│   │   └── densenet.py      # DenseNet CNN encoder (3 dense blocks)
│   └── decoder/
│       ├── __init__.py
│       ├── decoder.py       # SAN decoder with hierarchical attention and dual-path GRU
│       ├── attention.py     # Coverage-based attention mechanism
│       └── positional_encoding.py
├── training/                # Training pipeline
│   ├── __init__.py
│   ├── train.py             # Main training script (entry point)
│   ├── training.py          # Train/eval functions with metrics tracking
│   └── dataset.py           # HYBTr_Dataset class and dataloader creation
└── utils/                   # Model utilities
    ├── show_attention.py    # Attention visualization
    └── utils.py             # Config loading, checkpoint handling

data_tools/                  # Data processing utilities
├── dataset_prep/            # Dataset preparation scripts for CROHME
│   ├── prepare_crohme_data.py
│   ├── gen_hybrid_data.py
│   ├── gen_pkl.py
│   └── ...
├── synthetic_generator/     # Synthetic data generation system
│   ├── CLAUDE.md            # Synthetic generation documentation
│   ├── README.md
│   ├── scripts/             # Generation scripts
│   │   ├── generate_dataset.py
│   │   ├── synthesize_expression.py
│   │   └── ...
│   └── symbol_index.json
└── stroke_processing/       # Stroke-to-image conversion
    └── stroke2img.py

app/                         # Web application
├── backend/                 # Flask API server
│   ├── server.py            # HTTP server for stroke-to-LaTeX
│   └── inference_single.py  # Inference wrapper
└── frontend/                # Web UI

utils/                       # Shared utilities
└── Expression/              # LaTeX expression parsing
    └── gtd_parser.py        # GTD (Guided Tree Decoding) parser

data/                        # Datasets (train/test pickles)
checkpoints/                 # Saved model weights
logs/                        # TensorBoard logs
```

---

## Distinguishing Features

1. **Syntax-Aware Decoding**:
   - Explicitly models hierarchical structure of math expressions
   - Uses parent-child relationships to guide attention

2. **Coverage-Based Attention**:
   - Accumulates attention weights over timesteps
   - Prevents redundant attention to same image regions
   - Separate attention for each path

3. **Multi-Task Learning**:
   - Jointly predicts symbols (word_probs) and structures (struct_probs)
   - BCE loss for multi-label structure prediction

4. **Teacher Forcing with Tree Structure**:
   - During training, uses ground truth to retrieve parent hidden states
   - Maintains history of all parent states indexed by line ID
   - Enables efficient parallel processing of entire sequence

5. **Stack-Based Inference**:
   - Dynamically manages incomplete structural relations
   - Supports arbitrary nesting depth
   - No pre-defined maximum structure complexity

6. **Hierarchical GRU Architecture**:
   - Two-level GRU for both forward and backward paths
   - First level (input_gru) incorporates parent/partner information
   - Second level (out_gru) incorporates attention context

---

## Important Notes

- The model requires CUDA-compatible GPU (originally uses PyTorch 1.6.0+cu101)
- For Python 3.11+, use `requirements_python311.txt` for compatible package versions
- On macOS with Apple Silicon, PyTorch will use MPS (Metal Performance Shaders) instead of CUDA
- Pre-trained checkpoint available at `training/checkpoints/best.pth`
- Supports CROHME and MathWriting data sets
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
- **`../data_tools/synthetic_generator/CLAUDE.md`** - Complete pipeline documentation
- **`../data_tools/synthetic_generator/README.md`** - Quick start guide

The system generates training data by compiling LaTeX to DVI format, extracting glyph positions, and replacing symbols with real handwritten strokes from a library of 6,423 InkML files.
