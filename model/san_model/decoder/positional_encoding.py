import torch
import torch.nn as nn
import math


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for spatial attention mechanism.

    Generates sinusoidal position encodings for each spatial location (h, w)
    in the feature map, allowing the attention mechanism to be spatially aware.

    The encoding uses different frequencies for height and width dimensions:
    - PE[h, w, 2i]   = sin(h / 10000^(2i/d))
    - PE[h, w, 2i+1] = cos(h / 10000^(2i/d))
    - PE[h, w, d/2 + 2i]   = sin(w / 10000^(2i/d))
    - PE[h, w, d/2 + 2i+1] = cos(w / 10000^(2i/d))

    where d is the dimension of the positional encoding.
    """

    def __init__(self, d_model, max_height=100, max_width=100):
        """
        Args:
            d_model: Dimension of the positional encoding (should match attention_dim)
            max_height: Maximum height of feature map
            max_width: Maximum width of feature map
        """
        super(PositionalEncoding2D, self).__init__()

        self.d_model = d_model
        self.max_height = max_height
        self.max_width = max_width

        # Learnable projection to transform PE to attention dimension
        # This allows the model to learn how much each position dimension matters
        self.projection = nn.Linear(d_model, d_model)

        # Pre-compute the positional encoding for maximum dimensions
        # Will be sliced to actual dimensions during forward pass
        pe = self._generate_positional_encoding(max_height, max_width, d_model)

        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)

    def _generate_positional_encoding(self, height, width, d_model):
        """
        Generate 2D positional encoding.

        Args:
            height: Height of the feature map
            width: Width of the feature map
            d_model: Dimension of encoding

        Returns:
            Tensor of shape [height, width, d_model]
        """
        # Create position indices
        y_position = torch.arange(0, height).unsqueeze(1).float()  # [H, 1]
        x_position = torch.arange(0, width).unsqueeze(0).float()   # [1, W]

        # Expand to full spatial grid
        y_position = y_position.expand(height, width)  # [H, W]
        x_position = x_position.expand(height, width)  # [H, W]

        # Create encoding for y (height) dimension
        # Use first half of dimensions for height
        d_half = d_model // 2
        div_term = torch.exp(torch.arange(0, d_half, 2).float() *
                             -(math.log(10000.0) / d_half))

        pe = torch.zeros(height, width, d_model)

        # Encode height dimension (first half of channels)
        pe[:, :, 0:d_half:2] = torch.sin(y_position.unsqueeze(2) * div_term)
        pe[:, :, 1:d_half:2] = torch.cos(y_position.unsqueeze(2) * div_term)

        # Encode width dimension (second half of channels)
        pe[:, :, d_half::2] = torch.sin(x_position.unsqueeze(2) * div_term)
        pe[:, :, d_half+1::2] = torch.cos(x_position.unsqueeze(2) * div_term)

        return pe

    def forward(self, height, width):
        """
        Get positional encoding for given spatial dimensions.

        Args:
            height: Height of current feature map
            width: Width of current feature map

        Returns:
            Positional encoding tensor of shape [height, width, d_model]
        """
        # If dimensions exceed pre-computed maximum, generate new encoding
        if height > self.max_height or width > self.max_width:
            pe = self._generate_positional_encoding(height, width, self.d_model)
            pe = pe.to(self.pe.device)
        else:
            # Slice pre-computed encoding to actual dimensions
            pe = self.pe[:height, :width, :]

        # Apply learnable projection
        pe = self.projection(pe)

        return pe
