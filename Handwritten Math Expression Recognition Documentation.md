# Handwritten Math Expression Recognition {#handwritten-math-expression-recognition}

[Handwritten Math Expression Recognition](#handwritten-math-expression-recognition)

[Introduction](#introduction)

[Algorithm Design](#algorithm-design)

[Basis: Syntax-Aware Network](#basis:-syntax-aware-network)

[Adaptations](#adaptations)

[Attention](#attention)

[Regularisation](#regularisation)

[End-of-Sequence](#end-of-sequence)

[Loop Detection](#loop-detection)

[Token Priors](#token-priors)

[Data Preparation](#data-preparation)

[Data Sources](#data-sources)

[Stroke-to-Image Conversion](#stroke-to-image-conversion)

[Additional Constructs](#additional-constructs)

[Accents](#accents)

[Matrices, Vectors and Systems of Equations](#matrices,-vectors-and-systems-of-equations)

[Dutch Logarithms](#dutch-logarithms)

[Spaces](#spaces)

## Introduction {#introduction}

This documentation describes the Handwritten Math Expression Recognition (HMER) algorithm used in Algebrakit. The goal is to provide a comprehensive overview of the design choices of the algorithm and the data preparation pipeline. 

## Algorithm Design {#algorithm-design}

### Basis: Syntax-Aware Network {#basis:-syntax-aware-network}

This algorithm is based on the open-source implementation of Syntax-Aware Network ([Github](https://github.com/tal-tech/SAN), [Research Paper](https://arxiv.org/abs/2203.01601)). The core ideas of this approach are:

* **An Encoder-Decoder model with attention**  
  * The encoder uses DenseNet, a standard recurrent convolutional network.  
  * The decoder is a recurrent neural network with an attention module  
* **Using a math syntax tree.** The decoder does not directly predict the LaTeX expression. Such decoders do exist, but can generate invalid LaTeX. For example, unbalanced curly braces.   
  Instead, a math expression is represented as a tree of tokens and relations. There are seven possible relations: above, below, sup, sub, left-sup, right, and inside.   
  * The syntax of the tree is as follows:   
    `[<symbol>, <symbol id>,  <parent id>, <relation>]  `
  * Example: `\frac{2x^3}{x+1}`
    ```
    [‘<s>’, 0, -1, ‘root’]  
    [‘\frac’,1, 0, ‘Start’]  
    [‘2’, 2, 1, ‘above’]  
    [‘x’, 3, 2, ‘right’]  
    [‘3’, 4, 3, ‘sup’]  
    [‘x’, 5, 1, ‘below’]  
    [‘+’, 6, 5, ‘right’]  
    [‘1’, 7, 6, ‘right’]  
    ```
* The model uses a variation which immediately identifies all relations of a parent token. For example: token `\frac` has a `struct` element with types `[‘above’, ‘below’]`. The types are identified by a one-hot vector `[<above>, <below>, <sub>, <sup>, <L-sup>, <right>]`.  
  Symbols are concatenated left-to-right until an <eos> (‘End of Stroke’) symbol.  
  * Example: `\frac{2x^3}{x+1}`  
    ```
    [0, ‘\frac’, \-1, ‘<s>’]
    [1, ‘struct’, 0, ‘\frac’, ‘above’, ‘below’,’’,’’,’’,’’,’’]  
    [2, ‘2’, 1, ‘above’]  
    [3, ‘x’, 2, ‘2’]  
    [4, ‘struct’, 3, ‘’,’’,’’,’sup’,’’,’’,’’]  
    [5, ‘3’, 4, ‘sup’]  
    [6, ‘<eos>, 5, ‘3’]  
    [7, ‘x’, 2, ‘below’]  
    [8, ‘+’, 7, ‘x’]  
    [9, ‘1’, 8, ‘+’]  
    [10, <eos>, 9, ‘9’]
    ```

### Adaptations {#adaptations}

A number of changes were made to the original SAN model.

* The attention model is improved  
* The reversed regularisation model is removed  
* An extra loss function for the `<eos>` symbol is added  
* A safety mechanism is added to prevent infinite repetition of the same symbol.  
* A mechanism is added to influence token selection using pre-defined priors.

These changes are discussed below.

#### Attention {#attention}

The attention module computes a weighting vector to enable the decoder to focus on the most relevant part of the image. This weighting vector is based on three pieces of information:

* **The encoder output**  
* **The decoder’s inner state vector**, based on the previous state and the last generated token  
* **The sum of the past attention weight vectors.** The sum is taken over all the ‘parents’, which is all previous tokens going backwards (right-to-left) through the sequences and upwards through the parents.  
  For example, if the decoder has parsed `1 \+ \frac{2+x}{y^..}`, and is currently in the `sup` of `y`, the parents are `y`, `\frac`, `+`, `1`. The attention weight vector is the sum of the attention vectors of all these tokens. Note that the symbols in the sub-expression `2+x` are not included.

Experiments revealed that the attention weights become diffuse after a number of steps. This means the weights are applied to a large portion of the image, making the attention mechanism ineffective.  
The problem is that all previous attention weights are added. Instead, the previous attention weights should be divided into two classes: 

* Attention weights of symbols that are handled completely and are no longer relevant.  
* Attention weights of symbols that are relevant to compute the next token.

Example: Assume the decoder has parsed `1 + \frac{2+x}{a+..}`, and is currently in the `below` of `\frac`. Then, the relevant attention weights are from

1. the last generated symbol (‘+’), as we expect the next symbol to be to its right,   
2. the active parent `\frac` and its `struct`, which identifies the fraction area.

The area of the image corresponding to symbols `1` and `+` is not of interest anymore. 

The idea is that the ‘relevant’ attention weights must be added to **increase** the attention weights (and draw attention to these areas), while the ‘completed’ attention weights must be subtracted to **decrease** the attention weights (and prevent these areas from being considered again).

Instead of adding or subtracting, we will keep two attention-weight sums and apply a trained transformation matrix to each sum. Training will ensure that the ‘relevant’ sum increases attention and the ‘completed’ sum penalises attention.

The parent weights are added with exponential decay, such that a parent higher up the tree has less impact than the immediate parent. The decay factor is a trainable parameter (typically around 0.5).

This new attention mechanism shows much better focus throughout the decoder’s steps.

#### Regularisation {#regularisation}
The SAN model uses a reverse encoder, which works right-to-left and computes parents from childs. The idea is that the attention weights for a symbol should be similar when working forward or backward. 
The reason to delete it is that:
1. it increases the complexity of the model (the number of trainable parameters) by a factor of two. 
2. the new attention model works well without an extra regularisation module.

##### End-of-Sequence  {#end-of-sequence}
The `<eos>` token ends a left-to-right concatenation of symbols. An error related to this token (generating it while it shouldn't or not generating it while it should) has a global impact on the expression, breaking its structure. Therefore, it makes sense to put more weight on the accuracy of the `<eos>` token than the other word tokens.

#### Loop Prevention {#loop-detection}
Experiments showed that one-symbol expressions sometimes get stuck in an infinite loop. So instead of generating '6', it generates '6666666666...'. It turns out that this can be eliminated by ensuring the images are large enough to ensure the encoded vector is at least 2x2. This means the images must be at least 32x32 pixels. 

#### Token Priors {#token-priors}

### Model tuning

#### Encoder
The default implementation from the SAN article uses DenseNet with Bottleneck (DenseNet-B) with the following configuration:
- Growth rate $k=24$
- Initial $7\times 7$ convolution with stride 2, which means height and width are divided by 2.
- 2D Max Pooling, which reduces the height and width again by a factor of 2.
- Three dense blocks, with $n=16$ layers each. Each layer contains a $3\times 3$ convolution with $k$ output layers. The input is all channels of all previous layers, and the original input.
- A transition layer after the first two blocks halves the spatial dimensions and the number of channels. 
- Bottleneck (BNK): the $3\times 3$ convolution is prepended by a $1\times 1$ convolution that reduces the number of channels to $4k$ before entering the $3\times 3$ convolution.

In total, the size of the image is reduced by a factor of 16 (four times divided by 2). Each feature vector generated by DenseNet is roughly based on $32n\times 32n$ pixels, where $n$ is the number of layers in the (last) dense block.

| Component | Input Ch | Output Ch | Parameters |
| --- | --- | --- | --- |
| BNK1 | $l_0$ | $4k$ | $l_0\times 4k$|
| L1 | $4k$ | $k$ | $9\times 4k^2$|
| BNK2 | $l_0+k$ | $4k$ | $(l_0+k)\times 4k$|
| L2 | $4k$ | $k$ | $9\times 4k^2$|
| BNK3 | $l_0+2k$ | $4k$ | $(l_0+2k)\times 4k$|
| L3 | $4k$ | $k$ | $9\times 4k^2$|
| ... | ... | ... | |
| BNKn | $l_0+(n-1)\times k$ | $4k$ |$(l_0+(n-1)k)\times 4k$ |
| Ln | $4k$ | $k$ |$9\times 4k^2$ |

Total nr of output channels is $l_0 + nk$. 

Total nr of parameters of one dense block \
= $36nk^2 + 4l_0nk + 4k^2 \sum_{i=1}^{n}(i-1)$ \
= $36nk^2 + 4l_0nk + 4k^2\times \frac{1}{2}n(n-1)$ \
= $36nk^2 + 4l_0nk + 2k^2n(n-1)$ \
= $(34+2n)nk^2 + 4l_0nk$

Note that the complexity is quadratic in the number of layers and the growth rate. For stroke-based images, which don't contains noise or shadows, we think the original DenseNet configuration is too big.

| $k$ | $n$ | nr channels | nr parameters ($l_0=2k$) |
| --- | --- | ---| --- |
| 24 | 16 | $l_0+384$ | 682K |
| 16 | 16 | $l_0+256$ | 303K |
| 12 | 16 | $l_0+192$ | 170K |
| 16 | 12 | $l_0+192$ | 203K |
| 16 | 8 | $l_0+128$ | 119K |

The number of output channels of DenseNet is as follows:

| component | output channels |
| --- | --- | 
| initial convolution | $2k$ | 
| dense1 | $2k + nk$ |
| transition1 | $k+\frac12nk$ |
| dense2 | $k+\frac32nk$ |
| transition2 | $\frac12 k+\frac34nk$ |
| dense3 | $\frac12 k+\frac74nk$ |

| $k$ | $n$ | nr channels 
| --- | --- | ---| 
| 24 | 16 | $684$ 
| 16 | 12 | $344$ 
| 12 | 16 | $342$ 
| 16 | 8 | $232$ 

#### Considerations for choosing parameters
With $n=8$ layers, a feature vector covers a range of $32n=256$ pixels. Experiments indicate that this may be too small to detect the relations for a symbol. E.g. for an expression of the form $e^{expr}$ where the exponent is a large expression, the decoder needs to decide when it parsed the $e$, whether there will be a `right` relation or not. For this, it needs to be able to look past the exponent. Parsing matrices might also require a large support per feature vector.

Therefore, we keep $n=16$. However, the value for $k$ can be reduced from $k=24$ to $k=12$


#### Optimiser


## Data Preparation {#data-preparation}
The following tools are available to prepare a data set:
- `filter_labels.py`: 
  - cleans expressions: remove font styles, non-math commands (\mathbf, \textrm, etc), 
  - convert matrix notations to `\stack` syntax.
  - filters out expressions with unsupported symbols, like `\bigoplus`, `\bigvee`, etc
  - normalises the latex expression by parsing the expression and serialising again. This removes any meaningless variations in the latex. 
- `prep_latex.py`: Converts expressions into token lists
  - tokens are symbols (`1`, `x`), commands (`\frac`) and syntax symbols (`{`, `_`)
  - tokens are separated by space
- `inkml_to_images.py`: Convert InkML (strokes) into bitmap images.
- `split_dataset.py`: Split a dataset into a train, test and validation set.
- `gen_hybrid_data`: Generates the math syntax trees from latex expressions
- `gen_symbols_struct_dict`: Generates the list of tokens into a `word.txt` file.
- `gen_pkl.py`: Generates the Python pkl files containing images and labels for the test and training set.

### Data Sources {#data-sources}

### Stroke-to-Image Conversion {#stroke-to-image-conversion}

### Additional Constructs {#additional-constructs}

#### Math Accents {#accents}
Math Accents are constructs like `\hat{}`, `\bar{}` and `\underline`. These are encoded as a command token with a subexpression below or above it. 

For example, `\hat{x}` is encoded as 
```
[0, ‘\hat’, \-1, ‘<s>’]
[1, ‘struct’, 0, ‘\hat, '', ‘below’,’’,’’,’’,’’,’’]  
[2, ‘x’, 1, 'below']  
[3, <eos>, 2, ‘x’]
```
#### Matrices, Vectors and Systems of Equations {#matrices,-vectors-and-systems-of-equations}
Matrics and systems of equations use a vertical stack of expressions. In matrics there can be a horizontal alignment of expressions in columns. This two-dimensional structure of matrices does not fit the left-to-right and top-down approach of the algorithm, so it will struggle in large matrics to decide in which column a symbol belongs. 


#### Dutch Logarithms {#dutch-logarithms}
In the Netherlands (and some other countries such as Indonesia), the base of the logarithm is written to the top-left of the log token. This notation is encoded as non-standard latex `\lognl[2](x)`, which renders as ${}^2\log(x)$.

We don't want to introduce a `\lognl` token while training, as the command looks exactly the same as the `\log` token. This will be handled when converting from latex to hybrid and back.

#### Spaces {#spaces}
Spaces are default ignored, so latex expression `x\ =\ 2` is normalised into token list `x,=,2`.
This leads to problems for expressions with natural language. 

Examples:
- `x=2\ or\ x=3`: Normalising would give `x,2,o,r,x,=,3` which in latex is `x=2orx=3`
- `15 euro per minute`. Normalising would combine the three words into one.

The space cannot be removed if the symbols before and after it are alphabetic characters. In that case, we will use a `\space` token. If we include examples with words in the training set, the model should be able to capture the concept of words and learn to keep spaces, even for words it has not seen before.



