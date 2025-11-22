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
* An extra loss function for the <eos> symbol is added  
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

This new attention mechanism shows much better focus throughout the decoder’s steps.

#### Regularisation {#regularisation}
The SAN model uses a reverse encoder, which works right-to-left and computes parents from childs. The idea is that the attention weights for a symbol should be similar when working forward or backward. 
The reason to delete it is that:
1. it increases the complexity of the model (the number of trainable parameters) by a factor of two. 
2. the new attention model works well without an extra regularisation module.

##### End-of-Sequence  {#end-of-sequence}
The `<eos>` token ends a left-to-right concatenation of symbols. An error related to this token (generating it while it shouldn't or not generating it while it should) has a global impact on the expression, breaking its structure. Therefore, it makes sense to put more weight on the accuracy of the '<eos>' token than the other word tokens.

#### Loop Detection {#loop-detection}

#### Token Priors {#token-priors}

## Data Preparation {#data-preparation}

### Data Sources {#data-sources}

### Stroke-to-Image Conversion {#stroke-to-image-conversion}

### Additional Constructs {#additional-constructs}

#### Accents {#accents}

#### Matrices, Vectors and Systems of Equations {#matrices,-vectors-and-systems-of-equations}

#### Dutch Logarithms {#dutch-logarithms}

#### 

#### Spaces {#spaces}

