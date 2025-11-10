# Backlog

## Handle \lognl
Create a construct for Log which supports L-sup

## word operators

### standard words
Add standard words to the dataset for all languages.
- NL: of, en
- EN: or, and
- LT: arba, lt
- FR: e, ou
- DE: und, oder
- SP: ..

### word spacing
To support expressions as: x=2 or x=3, the space after 'or' is important. 
So we must add a space symbol and create the following label: x = 2 o r \space x = 3
So the space is used only to separate words
Other examples
- 12 minuten rijden --> 1 2 m i n u t e n \space r i j d e n

### word concept 
Add inputs with words to the dataset. This allows the model to understand the concept of a word (e.g. that no digits or math symbols exist in them and that spacing is important)

## add symbols
We lack data for some symbols:
- money: €, $, £

## Data collection tool 
For data collection from real people, it is convenient to have a data collection tool

## Improve support for matrices
Find a way to better support matrices of dimension > 2

