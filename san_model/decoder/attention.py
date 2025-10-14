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

        # Note: the linear transformation can be absorbed in the convolution as it acts on the channel dimension only
        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, padding=5, bias=False)
        self.attention_weight = nn.Linear(512, self.attention_dim, bias=False)
        self.alpha_convert = nn.Linear(self.attention_dim, 1)

        # Learnable weights to balance attention terms
        query_weight_init = params.get('attention', {}).get('query_weight', 1.0)
        coverage_weight_init = params.get('attention', {}).get('coverage_weight', 3.0)
        features_weight_init = params.get('attention', {}).get('features_weight', 1.0)

        self.query_weight = nn.Parameter(torch.tensor(query_weight_init))
        self.coverage_weight = nn.Parameter(torch.tensor(coverage_weight_init))
        self.features_weight = nn.Parameter(torch.tensor(features_weight_init))

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
            # Learnable weight for position encoding (same as other terms)
            position_weight_init = params.get('attention', {}).get('position_encoding_weight', 2.0)
            self.position_weight = nn.Parameter(torch.tensor(position_weight_init))

    def forward(self, cnn_features, hidden, alpha_sum, image_mask=None):

        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        # Note: the linear transformation can be absorbed in the convolution as it acts on the channel dimension only
        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0,2,3,1))

        cnn_features_trans = self.encoder_feature_conv(cnn_features)

        # Get spatial dimensions
        height, width = cnn_features_trans.shape[2:]

        # Apply learnable weights to each term and combine
        weighted_query = self.query_weight * query[:, None, None, :]
        weighted_coverage = self.coverage_weight * coverage_alpha
        weighted_features = self.features_weight * cnn_features_trans.permute(0,2,3,1)

        # Add positional encoding if enabled
        if self.use_position_encoding:
            pos_encoding = self.position_encoding(height, width)  # [H, W, attention_dim]
            pos_encoding = pos_encoding.unsqueeze(0)  # [1, H, W, attention_dim]
            weighted_position = self.position_weight * pos_encoding
            alpha_score = torch.tanh(weighted_query + weighted_coverage + weighted_features + weighted_position)
        else:
            alpha_score = torch.tanh(weighted_query + weighted_coverage + weighted_features)

        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = torch.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)

        alpha_sum = alpha[:,None,:,:] + alpha_sum

        context_vector = (alpha[:,None,:,:] * cnn_features).sum(-1).sum(-1)

        return context_vector, alpha, alpha_sum
