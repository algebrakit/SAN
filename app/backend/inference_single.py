import torch

# import sys
# sys.path.append('../..') # only for local testing
from utils.Expression.gtd_parser import parse_gtd
from model.utils.utils import load_config, load_checkpoint
from model.inference.Backbone import Backbone
from model.training.dataset import Words
from utils.Expression.base import LatexOptions
from latex_utils import expand_symbol_adjustments

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

    def _build_priors_tensor(self, symbol_adjustments):
        """Convert symbol adjustments to log priors tensor.

        Args:
            symbol_adjustments: List of dicts with 'symbol' and 'offset' keys.
                               offset is one of: DISABLE, PENALIZE, BOOST, STRONG_BOOST
                               Symbols can be combined (e.g., 'BD', 'A_{0}') and will be
                               expanded to their basic symbols.

        Returns:
            torch.Tensor of shape (1, word_num) with log prior offsets, or None if no adjustments.

        Raises:
            ValueError: If offset type is invalid.
        """
        if not symbol_adjustments:
            return None

        # Expand combined symbols (e.g., 'BD' -> 'B', 'D') and filter to vocabulary
        vocabulary = set(self.params['words'].words_index_dict.values())
        symbol_adjustments = expand_symbol_adjustments(symbol_adjustments, vocabulary)

        if not symbol_adjustments:
            return None

        offsets = self.params['decoder'].get('prior_offsets', {
            'DISABLE': -100,
            'PENALIZE': -3,
            'BOOST': 2,
            'STRONG_BOOST': 4
        })

        device = self.params['device']
        priors = torch.zeros(1, self.params['word_num']).to(device=device)

        for adj in symbol_adjustments:
            symbol = adj['symbol']
            offset_type = adj['offset']

            # Validate offset type
            if offset_type not in offsets:
                raise ValueError(f"Invalid offset type: '{offset_type}'. Must be one of: {list(offsets.keys())}")

            # Validate symbol exists in vocabulary
            if symbol in self.params['words'].words_index_dict.values():
                idx = self.params['words'].encode([symbol])[0]
                priors[0, idx] = offsets[offset_type]
            else:
                pass
                # raise ValueError(f"Unknown symbol: '{symbol}'")


        return priors

    def convert2latex(self, img, symbol_adjustments=None):
        with torch.no_grad():
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = torch.Tensor(img) / 255
            image = image.unsqueeze(0).unsqueeze(0)

            image_mask = torch.ones(image.shape)
            device = self.params['device']
            image, image_mask = image.to(device), image_mask.to(device)

            # Build per-request priors tensor from symbol adjustments
            word_log_priors = self._build_priors_tensor(symbol_adjustments)

            prediction = self.model(image, image_mask, word_log_priors)
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