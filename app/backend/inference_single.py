import torch

# import sys
# sys.path.append('../..') # only for local testing
from utils.Expression.gtd_parser import parse_gtd
from model.utils.utils import load_config, load_checkpoint
from model.inference.Backbone import Backbone
from model.training.dataset import Words
from utils.Expression.base import LatexOptions

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

    def convert2latex(self, img):
        with torch.no_grad():
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = torch.Tensor(img) / 255
            image = image.unsqueeze(0).unsqueeze(0)

            image_mask = torch.ones(image.shape)
            device = self.params['device']
            image, image_mask = image.to(device), image_mask.to(device)

            prediction = self.model(image, image_mask)
            expr = parse_gtd(prediction)
            if expr is None:
                return None
            else:
                options = LatexOptions(convertLog=True, convertMatrices=True)
                latex_string = expr.toLatex(options)
            # latex_list = self.convert(1, prediction)
            # latex_string = ' '.join(latex_list)
            print('prediction=', prediction, 'latex_string=', latex_string)
            return latex_string

# inf = Inference()
# img = cv2.imread('/Users/martijnslob/Downloads/0b51625937d5ea2e.bmp', cv2.IMREAD_GRAYSCALE)
# latex = inf.convert2latex(img)
# print(latex)