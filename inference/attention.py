import torch
import torch.nn as nn


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

    def forward(self, cnn_features, hidden, alpha_sum, image_mask=None):
        # the hidden state c^{\alpha}_0 which combines the historic state and the latest terminal symbol or relation
        query = self.hidden_weight(hidden)
        # alpha_sum is 2D vector over encoded features. Use convolution to change nr. of channels to 'attention_dim' 
        # and keep 2D geometrical orientation of rows and colums in the image
        alpha_sum_trans = self.attention_conv(alpha_sum)
        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0,2,3,1))

        # change #channels from dimension of feature vector to dimension ('out_channels') of attention vector ('attention_dim')
        cnn_features_trans = self.encoder_feature_conv(cnn_features)

        # attention score dims: 1 x features rows x features cols x attention_dim
        alpha_score = torch.tanh(query[:, None, None, :] + coverage_alpha + cnn_features_trans.permute(0,2,3,1))
        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = torch.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        alpha_sum = alpha[:,None,:,:] + alpha_sum

        # inner product of weights (alpha) and features (oriented in 2D row/column tensor)
        context_vector = (alpha[:,None,:,:] * cnn_features).sum(-1).sum(-1)

        # energy_coverage = self.alpha_convert(torch.tanh(coverage_alpha))
        energy_coverage = self.alpha_convert(coverage_alpha)
        # energy_coverage = energy_coverage - energy_coverage.max()
        # energy_coverage_exp = torch.exp(energy_coverage.squeeze(-1))
        # if image_mask is not None:
        #     energy_coverage_exp = energy_coverage_exp * image_mask.squeeze(1)
        # alpha_coverage = energy_coverage_exp / (energy_coverage_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        alpha_coverage = energy_coverage
        
        # alpha_query = torch.tanh(query[:, None, None, :] + cnn_features_trans.permute(0,2,3,1))
        alpha_query = query[:, None, None, :] + cnn_features_trans.permute(0,2,3,1)
        energy_query = self.alpha_convert(alpha_query)
        # energy_query = energy_query - energy_query.max()
        # energy_query_exp = torch.exp(energy_query.squeeze(-1))
        # if image_mask is not None:
        #     energy_query_exp = energy_query_exp * image_mask.squeeze(1)
        # alpha_query = energy_query_exp / (energy_query_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        alpha_query = energy_query
        return context_vector, alpha, alpha_sum, alpha_query, alpha_coverage
