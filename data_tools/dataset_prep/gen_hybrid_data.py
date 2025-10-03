import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hybrid_to_latex import hybrid_to_latex
from utils.Expression.expression import Expression

class Tree:
    def __init__(self, label, parent_label='None', id=0, parent_id=0, op='none', brackets_open=0):
        self.children = []
        self.label = label
        self.id = id
        self.parent_id = parent_id
        self.parent_label = parent_label
        self.op = op
        self.brackets_open = brackets_open # number of '{' opened when this node is created

def process_folder(fnameIn:str, folderOut:str):
    with open(fnameIn) as f:
        lines = f.readlines()

    filtered = []

    for line in tqdm(lines):
        # line = 'RIT_2014_178.jpg x ^ { \\frac { p } { q } } = \sqrt [ q ] { x ^ { p } } = \sqrt [ q ] { x ^ { p } }'
        name, *words = line.split()
        name = name.split('.')[0]

        latex = ' '.join(words)
        exp = Expression.fromLatex(latex)
        if not exp:
            print(f"Failed to parse LaTeX in {name}")
            sys.exit(1)

        hybrid = exp.to_hybrid()

        newlatex = hybrid_to_latex(1, hybrid)
        check = ' '.join(words) == newlatex
        if not check:
            newlatex = hybrid_to_latex(1, hybrid, True)
            check = ' '.join(words) == newlatex
            if not check:
                print(f"Mismatch in {name}")
                print('Original:', ' '.join(words))
                print('New.    :', newlatex)
                print(f"Skipping")
                continue

        filtered.append(line)
        with open(f'{folderOut}/{name}.txt', 'w') as f:
            for line in hybrid:
                f.write(' '.join(map(str, line))+'\n')

    with open(f'{fnameIn}.filtered', 'w') as f:
        for line in filtered:
            f.write(line)


# label = '/Users/martijnslob/github/SAN/data_tools/dataset_prep/train/train-test.txt'
# out = '/Users/martijnslob/github/SAN/data_tools/dataset_prep/train_hyb'
def main():
    if len(sys.argv) != 3:
        print("Usage: python gen_hybrid_data.py <train_file> <output_folder>")
        print("Example: python gen_hybrid_data.py train.txt output_folder")
        sys.exit(1)

    input_file = sys.argv[1]
    output_folder = sys.argv[2]

    try:
        process_folder(input_file, output_folder)
        print(f"Successfully filtered {input_file} -> {output_folder}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()