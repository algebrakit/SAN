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
        # self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, padding=5, bias=False)
        # self.attention_weight = nn.Linear(512, self.attention_dim, bias=False)
        self.attention_completed_conv = nn.Conv2d(1, self.attention_dim, kernel_size=11, padding=5, bias=False)
        self.attention_active_conv = nn.Conv2d(1, self.attention_dim, kernel_size=11, padding=5, bias=False)
        self.alpha_convert = nn.Linear(self.attention_dim, 1)

        # Learnable weights to balance attention terms
        query_weight_init = params.get('attention', {}).get('query_weight', 1.0)
        coverage_completed_weight_init = params.get('attention', {}).get('coverage_completed_weight', 3.0)
        coverage_active_weight_init = params.get('attention', {}).get('coverage_active_weight', 2.0)
        features_weight_init = params.get('attention', {}).get('features_weight', 1.0)

        self.query_weight = nn.Parameter(torch.tensor(query_weight_init))
        self.coverage_completed_weight = nn.Parameter(torch.tensor(coverage_completed_weight_init))
        self.coverage_active_weight = nn.Parameter(torch.tensor(coverage_active_weight_init))
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

    def forward(self, cnn_features, hidden, alpha_sum_completed, alpha_sum_active, image_mask=None, return_debug=False):
        """
        Compute attention with dual coverage aggregates.

        Args:
            cnn_features: Encoded image features
            hidden: Current decoder hidden state
            alpha_sum_completed: Sum of attention for completed symbols (penalize re-attending)
            alpha_sum_active: Sum of attention for active parent symbols (boost nearby attention)
            image_mask: Mask for variable-width images
            return_debug: If True, return intermediate values for visualization

        Returns:
            context_vector, alpha, alpha_sum_completed, alpha_sum_active
            (if return_debug=True, also returns query_features, coverage_combined)
        """
        query = self.hidden_weight(hidden)
        cnn_features_trans = self.encoder_feature_conv(cnn_features)

        # Get spatial dimensions
        height, width = cnn_features_trans.shape[2:]

        # Apply learnable weights to each term
        weighted_query = self.query_weight * query[:, None, None, :]
        weighted_features = self.features_weight * cnn_features_trans.permute(0,2,3,1)

        # Process completed coverage (penalize)
        # alpha_sum_completed = alpha_sum_completed / (alpha_sum_completed.sum(dim=(1,2,3), keepdim=True) + 1e-10) # normalise needed as the sum() equals the number of completed symbols
        alpha_completed_trans = self.attention_completed_conv(alpha_sum_completed)
        coverage_completed = alpha_completed_trans.permute(0,2,3,1)
        weighted_coverage_completed = self.coverage_completed_weight * coverage_completed

        # Process active coverage (boost)
        # alpha_sum_active = alpha_sum_active / (alpha_sum_active.sum(dim=(1,2,3), keepdim=True) + 1e-10) # normalise needed as the sum() equals the number of active parents
        alpha_active_trans = self.attention_active_conv(alpha_sum_active)
        coverage_active = alpha_active_trans.permute(0,2,3,1)
        weighted_coverage_active = self.coverage_active_weight * coverage_active

        # Combine terms: 
        alpha_score = weighted_query + weighted_features + weighted_coverage_completed + weighted_coverage_active

        # Add positional encoding if enabled
        if self.use_position_encoding:
            pos_encoding = self.position_encoding(height, width)  # [H, W, attention_dim]
            pos_encoding = pos_encoding.unsqueeze(0)  # [1, H, W, attention_dim]
            weighted_position = self.position_weight * pos_encoding
            alpha_score = alpha_score + weighted_position

        alpha_score = torch.tanh(alpha_score)

        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = torch.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        alpha = alpha[:,None,:,:]
        context_vector = (alpha * cnn_features).sum(-1).sum(-1)

        # Return attention map and current aggregates (for updating history)
        # Note: aggregates are updated externally in the decoder
        if return_debug:
            # For visualization: return query+features and coverage (completed-active)
            query_features = weighted_query + weighted_features + weighted_position
            coverage_combined = weighted_coverage_active + weighted_coverage_completed
            # Convert to 2D by taking the alpha channel (before softmax)
            query_features_2d = self.alpha_convert(torch.tanh(query_features)).squeeze(-1).squeeze(0)
            query_features_2d = self.to_softmax(query_features_2d, image_mask)
            coverage_combined_2d = self.alpha_convert(torch.tanh(coverage_combined)).squeeze(-1).squeeze(0)
            # coverage_combined_2d = self.to_softmax(coverage_combined_2d, image_mask)
            return context_vector, alpha, query_features_2d, coverage_combined_2d
        else:
            return context_vector, alpha

    def to_softmax(self, A, image_mask):
        energy = A - A.max()
        energy_exp = torch.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        return alpha
