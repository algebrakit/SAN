import torch.nn as nn
import sys
sys.path.append('..')
from san_model.encoder.densenet import DenseNet
from .san_decoder import SAN_decoder

class Backbone(nn.Module):
    def __init__(self, params=None):
        super(Backbone, self).__init__()

        self.params = params
        self.use_label_mask = params['use_label_mask']

        if params['encoder']['net'] == 'DenseNet':
            self.encoder = DenseNet(params=self.params)
        else:
            raise NotImplementedError(f"Encoder {params['encoder']['net']} not implemented")
        self.decoder = SAN_decoder(params=self.params)
        self.ratio = params['densenet']['ratio'] if params['encoder']['net'] == 'DenseNet' else 16 * params['resnet'][
            'conv1_stride']

    def forward(self, images, images_mask):

        cnn_features = self.encoder(images)
        prediction = self.decoder(cnn_features, images_mask, images)

        return prediction


