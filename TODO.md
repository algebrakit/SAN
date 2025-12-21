# Backlog

## Improve some math constructs
Lognl werkt nog niet goed
Accentnotatie bij expressies: ( 2x(x^2+1))’

## Add symbols
We lack data for some symbols:
- money: €, $, £
- \ddot, \dddot

## Support incomplete expressions
- 2=
- y=4+

## Frontend component
Mathquil: \root, \left{,
Autoscroll op mobiel

## PARTIALLY DONE: Priors
- Allow setting symbol priors
  
later:
- Pre-defined prior setting from syntax rules
  - interval notation
  - log (might be difficult)
  - non-standard Greek symbols
  - advanced math symbols

## DONE: Handle \lognl
Create a construct for Log which supports L-sup

## DONE: Words

### DONE: Standard words
Add standard words to the dataset for all languages.
- NL: of, en
- EN: or, and
- LT: arba, lt
- FR: e, ou
- DE: und, oder
- SP: ..

### DONE: Word spacing
To support expressions as: x=2 or x=3, the space after 'or' is important. 
So we must add a space symbol and create the following label: x = 2 o r \space x = 3
So the space is used only to separate words
Other examples
- 12 minuten rijden --> 1 2 m i n u t e n \space r i j d e n

Rule: 
- in dataset, transform a spacing (\ , \;, etc) to \space if:
   - the spacing is after a letter symbol (a-zA-Z)
   - and it followd by a letter or digit (a-zA-Z0-9)


### DONE: Word concept 
Add inputs with words to the dataset. This allows the model to understand the concept of a word (e.g. that no digits or math symbols exist in them and that spacing is important)


## DONE: Loop protection
- covered by enforcing image size to be at least 2x16px

## Data collection tool 
For data collection from real people, it is convenient to have a data collection tool

## Improve support for matrices
Find a way to better support matrices of dimension > 2

