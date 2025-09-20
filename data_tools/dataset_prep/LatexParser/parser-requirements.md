# A latex parser for normalising latex expressions
For to prepare a deep learning algorithm for handwriting recognition of math expressions, we need a functionality that read latex expressions and normalised them to a standard notations.
Examples:
-  Remove synonyms of commands:
   -  Input: {x+1}\over{4}, Output: \frac{x+1}{4}
-  Remove unneeded brackets
   -  Input: {n}, Output: n
   -  Input: \frac{ {x+2}n }{2x}, Output: \frac{ x+2n }{2x}
- But do apply brackets for commands
   -  Input: \frac12, Output: \frac{1}{2}
   -  Input: \sqrt[n]3, Output: \sqrt[n]{3}
-  Remove styling commands:
   -  Input: \mathbf{1+x}, Output: 1+x
   -  Input: \frac{\scriptstyle{WF}}{2}, Output: \frac{WF}{2}

## Interface
The function accepts a latex string, representing a single math expression.
The function returns the normalised latex string

## Implementation aspects
To properly handle commands and brackets, the latex expression must be parsed into a parse tree.
The normalised latex expression is generated from the parse tree.