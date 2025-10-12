import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding2D


class Attention(nn.Module):

    def __init__(self, params):
        super(Attention, self).__init__()

        self.params = params
        self.channel = params['encoder']['out_channels']
        self.hidden = params['decoder']['hidden_size']
        self.attention_dim = params['attention']['attention_dim']

        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim)
        self.encoder_feature_conv = nn.Conv2d(self.channel, self.attention_dim, kernel_size=1)

        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, padding=5, bias=False)
        self.attention_weight = nn.Linear(512, self.attention_dim, bias=False)
        self.alpha_convert = nn.Linear(self.attention_dim, 1)

        # Positional encoding for spatial awareness
        self.use_position_encoding = params.get('attention', {}).get('use_position_encoding', True)
        if self.use_position_encoding:
            # Calculate max spatial dimensions based on image size and downsampling ratio
            ratio = params['densenet']['ratio'] if params['encoder']['net'] == 'DenseNet' else 16 * params['resnet']['conv1_stride']
            max_height = params['image_height'] // ratio
            max_width = params['image_width'] // ratio

            self.position_encoding = PositionalEncoding2D(
                d_model=self.attention_dim,
                max_height=max_height,
                max_width=max_width
            )
            self.position_weight = params.get('attention', {}).get('position_encoding_weight', 1.0)

    def forward(self, cnn_features, hidden, alpha_sum, image_mask=None):

        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0,2,3,1))

        cnn_features_trans = self.encoder_feature_conv(cnn_features)

        # Get spatial dimensions
        height, width = cnn_features_trans.shape[2:]

        # Add positional encoding if enabled
        if self.use_position_encoding:
            pos_encoding = self.position_encoding(height, width)  # [H, W, attention_dim]
            pos_encoding = pos_encoding.unsqueeze(0)  # [1, H, W, attention_dim]
            alpha_score = torch.tanh(query[:, None, None, :] + coverage_alpha +
                                    cnn_features_trans.permute(0,2,3,1) +
                                    self.position_weight * pos_encoding)
        else:
            alpha_score = torch.tanh(query[:, None, None, :] + coverage_alpha +
                                    cnn_features_trans.permute(0,2,3,1))

        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = torch.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)

        alpha_sum = alpha[:,None,:,:] + alpha_sum

        context_vector = (alpha[:,None,:,:] * cnn_features).sum(-1).sum(-1)

        return context_vector, alpha, alpha_sum
