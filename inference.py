import os
import cv2
import argparse
import torch
import json
from tqdm import tqdm

from utils.Expression.gtd_parser import parse_gtd
from utils.utils import load_config, load_checkpoint
from inference.Backbone import Backbone
from training.dataset import Words

parser = argparse.ArgumentParser(description='Spatial channel attention')
parser.add_argument('--config', default='config.yaml', type=str, help='config file path')
parser.add_argument('--image_path', default='/home/yuanye/work/data/CROHME2014/14_off_image_test', type=str, help='test image path')
parser.add_argument('--label_path', default='/home/yuanye/work/data/CROHME2014/test_caption.txt', type=str, help='test label path')
args = parser.parse_args()

if not args.config:
    print('Please provide config yaml path!')
    exit(-1)

"""Load config file"""
params = load_config(args.config)

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = torch.device('mps')
else:
    device = torch.device('cpu')
params['device'] = device
print('Device used:', device)
words = Words(params['word_path'])
params['word_num'] = len(words)
params['struct_num'] = 7
params['words'] = words

model = Backbone(params)
model = model.to(device)

load_checkpoint(model, None, params['checkpoint'])

model.eval()

word_right, node_right, exp_right, length, cal_num = 0, 0, 0, 0, 0

with open(args.label_path) as f:
    labels = f.readlines()

with torch.no_grad():
    bad_case = {}
    count = 0
    for item in tqdm(labels):
        name, *label = item.split()
        label = ' '.join(label)
        #name = name + '_0.bmp'
        img = cv2.imread(os.path.join(args.image_path, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = torch.Tensor(img) / 255
        image = image.unsqueeze(0).unsqueeze(0)

        image_mask = torch.ones(image.shape)
        image, image_mask = image.to(device), image_mask.to(device)

        prediction = model(image, image_mask)
        expr = parse_gtd(prediction)
        if expr is None:
            latex_string = 'illegal'
        else:
            latex_string = expr.toLatex()

        # latex_list = convert(1, prediction)
        # latex_string = ' '.join(latex_list)
        if latex_string == label.strip():
            exp_right += 1
        else:
            bad_case[name] = {
                'label': label,
                'predi': latex_string,
                'list': prediction
            }

        count += 1
        if count % 100 == 0:
            print('Current ExpRate: ', exp_right / count)    
            with open('bad_case.json', 'w') as f:
                json.dump(bad_case, f, ensure_ascii=False)

        # break
    print(exp_right / len(labels))

with open('bad_case.json', 'w') as f:
    json.dump(bad_case, f, ensure_ascii=False)














