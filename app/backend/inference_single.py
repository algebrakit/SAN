import os
import cv2
import argparse
import torch
import json
from tqdm import tqdm

import sys
sys.path.append('../..')
from utils import load_config, load_checkpoint
from inference.Backbone import Backbone
from training.dataset import Words

class Inference:
    def __init__(self, confPath='config.yaml'):
        self.params = load_config(confPath)

        # Device selection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #     device = torch.device('mps')
        else:
            device = torch.device('cpu')
        self.params['device'] = device

        words = Words(self.params['word_path'])
        self.params['word_num'] = len(words)
        self.params['struct_num'] = 7
        self.params['words'] = words
        # structs = ['below', 'above', 'sub', 'sup', 'inside', 'L-sup', 'right']
        # struct_idx = words.encode(structs)
        # self.params['struct_dict'] = {structs[i]: struct_idx[i] for i in range(len(structs))}
        self.model = Backbone(self.params)
        self.model = self.model.to(device)

        load_checkpoint(self.model, None, self.params['checkpoint'])

        self.model.eval()

        word_right, node_right, exp_right, length, cal_num = 0, 0, 0, 0, 0

    def convert(self, nodeid, gtd_list):
        isparent = False
        child_list = []
        for i in range(len(gtd_list)):
            if gtd_list[i][2] == nodeid:
                isparent = True
                child_list.append([gtd_list[i][0],gtd_list[i][1],gtd_list[i][3]])
        if not isparent:
            return [gtd_list[nodeid][0]]
        else:
            if gtd_list[nodeid][0] == '\\frac':
                return_string = [gtd_list[nodeid][0]]
                for i in range(len(child_list)):
                    if child_list[i][2].lower() == 'above':
                        return_string += ['{'] + self.convert(child_list[i][1], gtd_list) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2].lower() == 'below':
                        return_string += ['{'] + self.convert(child_list[i][1], gtd_list) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2].lower() == 'right':
                        return_string += self.convert(child_list[i][1], gtd_list)
                for i in range(len(child_list)):
                    if child_list[i][2].lower() not in ['right','above','below']:
                        return_string += ['illegal']
            else:
                return_string = [gtd_list[nodeid][0]]
                for i in range(len(child_list)):
                    if child_list[i][2].lower() in ['l_sup']:
                        return_string += ['['] + self.convert(child_list[i][1], gtd_list) + [']']
                for i in range(len(child_list)):
                    if child_list[i][2].lower() == 'inside':
                        return_string += ['{'] + self.convert(child_list[i][1], gtd_list) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2].lower() in ['sub','below']:
                        return_string += ['_','{'] + self.convert(child_list[i][1], gtd_list) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2].lower() in ['sup','above']:
                        return_string += ['^','{'] + self.convert(child_list[i][1], gtd_list) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2].lower() == 'right':
                        return_string += self.convert(child_list[i][1], gtd_list)
            return return_string

    def convert2latex(self, img):
        with torch.no_grad():
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = torch.Tensor(img) / 255
            image = image.unsqueeze(0).unsqueeze(0)

            image_mask = torch.ones(image.shape)
            device = self.params['device']
            image, image_mask = image.to(device), image_mask.to(device)

            prediction = self.model(image, image_mask)
            latex_list = self.convert(1, prediction)
            latex_string = ' '.join(latex_list)
            print('prediction=', prediction, 'latex_string=', latex_string)
            return latex_string

# inf = Inference()
# img = cv2.imread('/Users/martijnslob/Downloads/0b51625937d5ea2e.bmp', cv2.IMREAD_GRAYSCALE)
# latex = inf.convert2latex(img)
# print(latex)