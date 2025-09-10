import os
import cv2
import argparse
import torch
import json
from tqdm import tqdm

import sys
sys.path.append('../..')
from san_model.utils import load_config, load_checkpoint
from inference.Backbone import Backbone
from training.dataset import Words

class Inference:
    def __init__(self, confPath='config.yaml'):
        self.params = load_config(confPath)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params['device'] = device

        words = Words(self.params['word_path'])
        self.params['word_num'] = len(words)
        self.params['struct_num'] = 7
        self.params['words'] = words

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
                    if child_list[i][2] == 'Above':
                        return_string += ['{'] + self.convert(child_list[i][1], gtd_list) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2] == 'Below':
                        return_string += ['{'] + self.convert(child_list[i][1], gtd_list) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2] == 'Right':
                        return_string += self.convert(child_list[i][1], gtd_list)
                for i in range(len(child_list)):
                    if child_list[i][2] not in ['Right','Above','Below']:
                        return_string += ['illegal']
            else:
                return_string = [gtd_list[nodeid][0]]
                for i in range(len(child_list)):
                    if child_list[i][2] in ['l_sup']:
                        return_string += ['['] + self.convert(child_list[i][1], gtd_list) + [']']
                for i in range(len(child_list)):
                    if child_list[i][2] == 'Inside':
                        return_string += ['{'] + self.convert(child_list[i][1], gtd_list) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2] in ['Sub','Below']:
                        return_string += ['_','{'] + self.convert(child_list[i][1], gtd_list) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2] in ['Sup','Above']:
                        return_string += ['^','{'] + self.convert(child_list[i][1], gtd_list) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2] in ['Right']:
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
            return latex_string
