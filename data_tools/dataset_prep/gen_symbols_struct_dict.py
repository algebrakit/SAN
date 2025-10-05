import os
import glob
import sys
from tqdm import tqdm

words_dict = set(['<eos>', '<sos>', 'struct'])

def process_folder(hyb_folder, words_file):
    labels = glob.glob(os.path.join(hyb_folder, '*.txt'))

    i = 3
    for item in tqdm(labels):
        with open(item) as f:
            lines = f.readlines()
        for line in lines:
            cid, c, pid, p, *r = line.strip().split()
            if c not in words_dict:
                words_dict.add(c)
                i+=1

    words_dict.remove('<eos>')
    words_dict.remove('<sos>')
    words_dict.remove('struct')
    words = [word for word in words_dict]
    words.sort()

    with open(words_file, 'w') as writer:
        writer.write('<eos>\n<sos>\nstruct\n')
        writer.write('\n'.join(words) + '\n')
        writer.write('above\nbelow\nsub\nsup\nL-sup\ninside\nright')

    print(i)


def main():
    if len(sys.argv) != 3:
        print("Usage: python gen_symbols_struct.py <hybrid_data_folder> <words_file>")
        print("Example: python gen_symbols_struct.py train_hyb word.txt")
        sys.exit(1)

    input_folder = sys.argv[1]
    words_file = sys.argv[2]

    try:
        process_folder(input_folder, words_file)
    except FileNotFoundError:
        print(f"Error: Input folder '{input_folder}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()