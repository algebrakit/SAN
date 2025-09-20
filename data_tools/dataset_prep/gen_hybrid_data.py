import os
import sys
from tqdm import tqdm

from hybrid_to_latex import hybrid_to_latex

class Tree:
    def __init__(self, label, parent_label='None', id=0, parent_id=0, op='none', brackets_open=0):
        self.children = []
        self.label = label
        self.id = id
        self.parent_id = parent_id
        self.parent_label = parent_label
        self.op = op
        self.brackets_open = brackets_open # number of '{' opened when this node is created


def convert(root: Tree, f):
    if root.tag == 'N-T':
        f.write(f'{root.id}\t{root.label}\t{root.parent_id}\t{root.parent_label}\t{root.tag}\n')
        for child in root.children:
            convert(child, f)
    else:
        f.write(f'{root.id}\t{root.label}\t{root.parent_id}\t{root.parent_label}\t{root.tag}\n')


position = set(['^', '_'])
math = set(['\\frac','\sqrt'])

def convertLine(words:list[str], name:str):
    # words = ['x', '^', '{', '\\frac', '{', 'p', '}', '{', 'q', '}', '=', '\\sqrt', '[', 'q', ']', '{', 'x', '^', '{', 'p', '}', '}', '=', '\\sqrt', '[', 'q', ']', '{', 'x', '^', '{', 'p', '}', '}']

    parents = []
    labels = []
    id = 1
    parents = [Tree('<sos>', id=0)]
    parent = Tree('<sos>', id=0)
    valid = True
    brackets_count = 0

    for i in range(len(words)):
        a = words[i]
        if a == '\\limits':
            continue
        if i == 0 and words[i] in ['_', '^', '{', '}']:
            print(name)
            valid = False
            break

        elif words[i] == '{':
            brackets_count += 1
            if words[i-1] == '\\frac':
                labels.append([id, 'struct', parent.id, parent.label])
                parents.append(Tree('\\frac', id=parent.id, op='above', brackets_open=brackets_count-1))
                id += 1
                parent = Tree('above', id=parents[-1].id+1)
            elif words[i-1] == '}' and brackets_count == parents[-1].brackets_open+1 and parents[-1].label == '\\frac' and parents[-1].op == 'above':
                parent = Tree('below', id=parents[-1].id+1)
                parents[-1].op = 'below'

            elif words[i-1] == '\sqrt':
                labels.append([id, 'struct', parent.id, '\sqrt'])
                parents.append(Tree('\sqrt', id=parent.id))
                parent = Tree('inside', id=id)
                id += 1
            elif words[i-1] == ']' and parents[-1].label == '\sqrt':
                parent = Tree('inside', id=parents[-1].id+1)

            elif words[i-1] == '^':
                if words[i-2] != '}':
                    if words[i-2] == '\sum' or words[i-2] == '\prod':
                        labels.append([id, 'struct', parent.id, parent.label])
                        parents.append(Tree(words[i-2], id=parent.id))
                        parent = Tree('above', id=id)
                        id += 1

                    else:
                        labels.append([id, 'struct', parent.id, parent.label])
                        parents.append(Tree(words[i-2], id=parent.id))
                        parent = Tree('sup', id=id)
                        id += 1

                else:
                    # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                    if parents[-1].label == '\sum' or parents[-1].label == '\prod':
                        parent = Tree('above', id=parents[-1].id+1)
                    else:
                        parent = Tree('sup', id=parents[-1].id + 1)
                    # id += 1

            elif words[i-1] == '_':
                if words[i-2] != '}':
                    if words[i-2] == '\sum' or words[i-2] == '\prod':
                        labels.append([id, 'struct', parent.id, parent.label])
                        parents.append(Tree(words[i-2], id=parent.id))
                        parent = Tree('below', id=id)
                        id += 1

                    else:
                        labels.append([id, 'struct', parent.id, parent.label])
                        parents.append(Tree(words[i-2], id=parent.id))
                        parent = Tree('sub', id=id)
                        id += 1

                else:
                    # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                    if parents[-1].label == '\sum' or parents[-1].label == '\prod':
                        parent = Tree('below', id=parents[-1].id+1)
                    else:
                        parent = Tree('sub', id=parents[-1].id+1)
                    # id += 1
            else:
                print('unknown word before {', name, i)
                valid = False
                break

        elif words[i] == '[' and words[i-1] == '\sqrt':
            labels.append([id, 'struct', parent.id, '\sqrt'])
            parents.append(Tree('\sqrt', id=parent.id))
            parent = Tree('L-sup', id=id)
            id += 1
        elif words[i] == ']' and parents[-1].label == '\sqrt':
            labels.append([id, '<eos>', parent.id, parent.label])
            id += 1

        elif words[i] == '}':
            brackets_count -= 1
            if words[i-1] != '}':
                labels.append([id, '<eos>', parent.id, parent.label])
                id += 1

            if i + 1 < len(words) and words[i+1] == '{' and parents[-1].label == '\\frac' and parents[-1].op == 'above':
                continue
            if i + 1 < len(words) and words[i + 1] in ['_', '^']:
                continue
            elif i + 1 < len(words) and words[i + 1] != '}':
                parent = Tree('right', id=parents[-1].id + 1)

            parents.pop()


        else:
            if words[i] in ['^', '_']:
                continue
            labels.append([id, words[i], parent.id, parent.label])
            parent = Tree(words[i],id=id)
            id += 1
    return valid, labels        

def get_lines(labels, parent_dict):
    lines = []
    for line in labels:
        id, label, parent_id, parent_label = line
        if label != 'struct':
            lines.append([id, label, parent_id, parent_label, None, None, None, None, None, None, None])
        else:
            tem = [id, label, parent_id, parent_label]
            tem = tem + ['above'] if 'above' in parent_dict[id] else tem + [None]
            tem = tem + ['below'] if 'below' in parent_dict[id] else tem + [None]
            tem = tem + ['sub'] if 'sub' in parent_dict[id] else tem + [None]
            tem = tem + ['sup'] if 'sup' in parent_dict[id] else tem + [None]
            tem = tem + ['L-sup'] if 'L-sup' in parent_dict[id] else tem + [None]
            tem = tem + ['inside'] if 'inside' in parent_dict[id] else tem + [None]
            tem = tem + ['right'] if 'right' in parent_dict[id] else tem + [None]
            lines.append(tem)
    if label != '<eos>':
        lines.append([id+1, '<eos>', id, label, None, None, None, None, None, None, None])

    return lines

def process_folder(fnameIn:str, folderOut:str):
    with open(fnameIn) as f:
        lines = f.readlines()

    newlines = []

    for line in tqdm(lines):
        # line = 'RIT_2014_178.jpg x ^ { \\frac { p } { q } } = \sqrt [ q ] { x ^ { p } } = \sqrt [ q ] { x ^ { p } }'
        name, *words = line.split()
        name = name.split('.')[0]

        valid, labels = convertLine(words, name)
        if not valid:
            print(f"Skipping invalid line in {name}")
            continue

        parent_dict = {0:[]}
        for i in range(len(labels)):
            parent_dict[i+1] = []
            parent_dict[labels[i][2]].append(labels[i][3])

        newlines = get_lines(labels, parent_dict)
        newlatex = hybrid_to_latex(1, newlines)
        check = ' '.join(words) == newlatex
        if not check:
            newlatex = hybrid_to_latex(1, newlines, True)
            check = ' '.join(words) == newlatex
            if not check:
                print(f"Mismatch in {name}")
                print('Original:', ' '.join(words))
                print(f"Skipping")
                continue

        with open(f'{folderOut}/{name}.txt', 'w') as f:
            for line in newlines:
                f.write(' '.join(map(str, line))+'\n')

    with open(f'{fnameIn}.filtered', 'w') as f:
        for line in newlines:
            f.write('\t'.join(map(str, line)) + '\n')


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