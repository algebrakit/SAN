import torch
import torch.nn as nn
import torch.nn.functional as F
import math
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
import warnings
>>>>>>> Stashed changes
=======
import warnings
>>>>>>> Stashed changes


class PointSequenceLSTM(nn.Module):
    """Process temporal sequences of points within each stroke"""
    
    def __init__(self, params):
        super(PointSequenceLSTM, self).__init__()
        self.input_size = 3  # (x, y, t) coordinates
        self.hidden_size = params.get('point_lstm_hidden', 128)
        self.num_layers = params.get('point_lstm_layers', 2)
        self.dropout = params.get('point_lstm_dropout', 0.2)
        
        # Point-level feature extraction
        self.point_encoder = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # LSTM for processing point sequences
        self.point_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Output dimension after bidirectional LSTM
        self.output_size = self.hidden_size * 2
        
    def forward(self, stroke_points, stroke_mask=None):
        """
        Args:
            stroke_points: [batch, max_points, 3] - (x, y, t) coordinates
            stroke_mask: [batch, max_points] - mask for valid points
        Returns:
            point_features: [batch, max_points, hidden_size*2] - point-level features
            stroke_summary: [batch, hidden_size*2] - aggregated stroke representation
        """
        batch_size, max_points, _ = stroke_points.shape
        
        # Point-level encoding
        point_features = self.point_encoder(stroke_points)  # [batch, max_points, hidden_size]
        
        # LSTM processing of point sequence
        lstm_out, _ = self.point_lstm(point_features)  # [batch, max_points, hidden_size*2]
        
        # Masked aggregation for stroke summary
        if stroke_mask is not None:
            mask_expanded = stroke_mask.unsqueeze(-1).expand_as(lstm_out)
            masked_features = lstm_out * mask_expanded
            
            # Sum over valid points and normalize
            stroke_summary = masked_features.sum(dim=1)  # [batch, hidden_size*2]
            valid_count = stroke_mask.sum(dim=1, keepdim=True).clamp(min=1)
            stroke_summary = stroke_summary / valid_count
        else:
            # Simple mean pooling if no mask
            stroke_summary = lstm_out.mean(dim=1)  # [batch, hidden_size*2]
        
        return lstm_out, stroke_summary


class StrokeRelationTransformer(nn.Module):
    """Model relationships between different strokes using Transformer"""
    
    def __init__(self, params):
        super(StrokeRelationTransformer, self).__init__()
        self.stroke_dim = params.get('point_lstm_hidden', 128) * 2  # From bidirectional LSTM
        self.num_heads = params.get('transformer_heads', 8)
        self.num_layers = params.get('transformer_layers', 3)
        self.feedforward_dim = params.get('transformer_ff_dim', 512)
        self.dropout = params.get('transformer_dropout', 0.2)
        self.max_strokes = params.get('max_strokes', 20)
        
        # Positional encoding for stroke order
        self.position_embedding = nn.Embedding(self.max_strokes, self.stroke_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.stroke_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=self.dropout,
            batch_first=True,
            activation='relu'
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
    def forward(self, stroke_features, stroke_mask=None):
        """
        Args:
            stroke_features: [batch, max_strokes, stroke_dim] - stroke-level features
            stroke_mask: [batch, max_strokes] - mask for valid strokes
        Returns:
            contextual_strokes: [batch, max_strokes, stroke_dim] - contextualized stroke features
        """
        batch_size, max_strokes, _ = stroke_features.shape
        
        # Add positional encoding for stroke order
        positions = torch.arange(max_strokes, device=stroke_features.device)
        pos_embed = self.position_embedding(positions).unsqueeze(0)  # [1, max_strokes, stroke_dim]
        stroke_features = stroke_features + pos_embed
        
        # Create attention mask for transformer (True = ignore, False = attend)
        if stroke_mask is not None:
            # Convert from valid mask (1=valid) to attention mask (True=invalid)
            attn_mask = ~stroke_mask.bool()  # [batch, max_strokes]
        else:
            attn_mask = None
        
        # Apply transformer
        if attn_mask is not None:
            # Transformer expects src_key_padding_mask: [batch, seq_len]
            contextual_strokes = self.transformer(stroke_features, src_key_padding_mask=attn_mask)
        else:
            contextual_strokes = self.transformer(stroke_features)
        
        return contextual_strokes


class SpatialFeatureGenerator(nn.Module):
    """Generate spatial feature maps from contextual stroke features"""
    
    def __init__(self, params):
        super(SpatialFeatureGenerator, self).__init__()
        self.stroke_dim = params.get('point_lstm_hidden', 128) * 2
        self.out_channels = params['encoder']['out_channels']  # 684 to match DenseNet
        self.feature_height = params.get('feature_height', 20)  # Match DenseNet output
        self.feature_width = params.get('feature_width', 100)   # Match DenseNet output
        
        # Project stroke features to intermediate dimension
        self.stroke_projection = nn.Sequential(
            nn.Linear(self.stroke_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512)
        )
        
        # Spatial position embeddings
        self.spatial_embed_h = nn.Parameter(
            torch.randn(1, 512, self.feature_height, 1) * 0.02
        )
        self.spatial_embed_w = nn.Parameter(
            torch.randn(1, 512, 1, self.feature_width) * 0.02
        )
        
        # CNN layers to generate final feature maps
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.out_channels, 1),  # Project to DenseNet channels
            nn.ReLU(inplace=True)
        )
        
        # Initialize spatial embeddings
        nn.init.normal_(self.spatial_embed_h, 0, 0.02)
        nn.init.normal_(self.spatial_embed_w, 0, 0.02)
        
    def forward(self, contextual_strokes, stroke_positions, stroke_masks=None):
        """
        Args:
            contextual_strokes: [batch, max_strokes, stroke_dim] - contextualized features
            stroke_positions: [batch, max_strokes, 4] - [center_x, center_y, width, height]
            stroke_masks: [batch, max_strokes] - mask for valid strokes
        Returns:
            spatial_features: [batch, out_channels, feature_height, feature_width]
        """
        batch_size, max_strokes, _ = contextual_strokes.shape
        
        # Project stroke features
        stroke_proj = self.stroke_projection(contextual_strokes)  # [batch, max_strokes, 512]
        
        # Initialize spatial canvas
        canvas = torch.zeros(
            batch_size, 512, self.feature_height, self.feature_width,
            device=contextual_strokes.device
        )
        
        # Place each stroke at its spatial location
        for batch_idx in range(batch_size):
            for stroke_idx in range(max_strokes):
                # Skip if stroke is padding
                if stroke_masks is not None and stroke_masks[batch_idx, stroke_idx] == 0:
                    continue
                
                # Get stroke position 
                center_x, center_y, width, height = stroke_positions[batch_idx, stroke_idx]
                
                # Skip empty strokes
                if center_x == 0 and center_y == 0 and width == 0 and height == 0:
                    continue
                
                # Map normalized coordinates to feature grid
                # Coordinates are in range [-2, 2] for width, [-0.5, 0.5] for height
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                grid_x = int((center_x + 2.0) * self.feature_width / 4.0)
                grid_y = int((center_y + 0.5) * self.feature_height / 1.0)
                
                # Clamp to valid range
                grid_x = max(0, min(self.feature_width - 1, grid_x))
                grid_y = max(0, min(self.feature_height - 1, grid_y))
                
                # Calculate stroke influence area (simple Gaussian-like)
                stroke_feat = stroke_proj[batch_idx, stroke_idx]  # [512]
                
                # Simple point placement (can be improved with Gaussian spreading)
                canvas[batch_idx, :, grid_y, grid_x] += stroke_feat
=======
=======
>>>>>>> Stashed changes
                # Fixed coordinate mapping with proper scaling
                grid_x_f = ((center_x + 2.0) / 4.0) * (self.feature_width - 1)
                grid_y_f = ((center_y + 0.5) / 1.0) * (self.feature_height - 1)
                
                # Clamp to valid range first, then convert to int
                grid_x_f = max(0, min(self.feature_width - 1, grid_x_f))
                grid_y_f = max(0, min(self.feature_height - 1, grid_y_f))
                
                grid_x = int(grid_x_f)
                grid_y = int(grid_y_f)
                
                # Calculate stroke size for Gaussian spreading
                stroke_feat = stroke_proj[batch_idx, stroke_idx]  # [512]
                stroke_width = max(0.5, width * self.feature_width / 4.0)  # Scale width to grid
                stroke_height = max(0.5, height * self.feature_height / 1.0)  # Scale height to grid
                
                # Gaussian spreading instead of single point
                sigma_x = max(1.0, stroke_width / 2.0)
                sigma_y = max(1.0, stroke_height / 2.0)
                
                # Apply Gaussian distribution around stroke center
                for dy in range(-2, 3):  # 5x5 kernel
                    for dx in range(-2, 3):
                        target_x = grid_x + dx
                        target_y = grid_y + dy
                        
                        # Check bounds
                        if 0 <= target_x < self.feature_width and 0 <= target_y < self.feature_height:
                            # Gaussian weight (ensure tensor operations)
                            dx_norm = torch.tensor(float(dx / sigma_x), device=canvas.device, dtype=canvas.dtype)
                            dy_norm = torch.tensor(float(dy / sigma_y), device=canvas.device, dtype=canvas.dtype)
                            weight = torch.exp(-0.5 * (dx_norm**2 + dy_norm**2))
                            canvas[batch_idx, :, target_y, target_x] += stroke_feat * weight
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
        
        # Add spatial position embeddings
        canvas = canvas + self.spatial_embed_h + self.spatial_embed_w
        
        # Generate final spatial feature maps
        spatial_features = self.spatial_conv(canvas)  # [batch, out_channels, H, W]
        
        return spatial_features


class StrokeAwareEncoder(nn.Module):
    """Complete stroke-aware encoder that outputs spatial features compatible with SAN decoder"""
    
    def __init__(self, params):
        super(StrokeAwareEncoder, self).__init__()
        self.params = params
        
        # Components
        self.point_lstm = PointSequenceLSTM(params)
        self.stroke_transformer = StrokeRelationTransformer(params)
        self.spatial_generator = SpatialFeatureGenerator(params)
        
        # Store dimensions for compatibility checks
        self.max_strokes = params.get('max_strokes', 20)
        self.max_points_per_stroke = params.get('max_points_per_stroke', 100)
        
    def forward(self, stroke_data, stroke_masks, stroke_positions):
        """
        Args:
            stroke_data: [batch, max_strokes, max_points, 3] - normalized stroke data
            stroke_masks: [batch, max_strokes, max_points] - point-level masks
            stroke_positions: [batch, max_strokes, 4] - stroke bounding boxes
        Returns:
            spatial_features: [batch, 684, 20, 100] - spatial feature maps (same as DenseNet)
        """
        batch_size, max_strokes, max_points, _ = stroke_data.shape
        
        # Process each stroke independently through LSTM
        stroke_features = []
        stroke_valid_mask = []
        
        for stroke_idx in range(max_strokes):
            # Extract data for this stroke
            stroke_points = stroke_data[:, stroke_idx, :, :]  # [batch, max_points, 3]
            point_mask = stroke_masks[:, stroke_idx, :]       # [batch, max_points]
            
            # Process through point LSTM
            _, stroke_summary = self.point_lstm(stroke_points, point_mask)  # [batch, hidden*2]
            
            stroke_features.append(stroke_summary)
            
            # Determine if stroke has any valid points
            stroke_has_points = point_mask.sum(dim=1) > 0  # [batch]
            stroke_valid_mask.append(stroke_has_points)
        
        # Stack stroke features and masks
        stroke_features = torch.stack(stroke_features, dim=1)  # [batch, max_strokes, hidden*2]
        stroke_valid_mask = torch.stack(stroke_valid_mask, dim=1).float()  # [batch, max_strokes]
        
        # Model inter-stroke relationships
        contextual_strokes = self.stroke_transformer(stroke_features, stroke_valid_mask)
        
        # Generate spatial feature maps
        spatial_features = self.spatial_generator(
            contextual_strokes, stroke_positions, stroke_valid_mask
        )
        
        return spatial_features


# Factory function for creating stroke encoders
def create_stroke_encoder(params):
    """Create appropriate stroke encoder based on configuration"""
    encoder_type = params.get('stroke_encoder_type', 'full')
    
    if encoder_type == 'full':
        return StrokeAwareEncoder(params)
    else:
        raise ValueError(f"Unknown stroke encoder type: {encoder_type}")


if __name__ == '__main__':
    # Test the stroke encoder
    params = {
        'encoder': {'out_channels': 684},
        'point_lstm_hidden': 128,
        'point_lstm_layers': 2,
        'transformer_heads': 8,
        'transformer_layers': 3,
        'max_strokes': 20,
        'max_points_per_stroke': 100,
        'feature_height': 20,
        'feature_width': 100
    }
    
    # Create test data
    batch_size = 2
    stroke_data = torch.randn(batch_size, 20, 100, 3)
    stroke_masks = torch.randint(0, 2, (batch_size, 20, 100)).float()
    stroke_positions = torch.randn(batch_size, 20, 4)
    
    # Create encoder
    encoder = StrokeAwareEncoder(params)
    
    # Test forward pass
    with torch.no_grad():
        output = encoder(stroke_data, stroke_masks, stroke_positions)
        print(f"Input shape: {stroke_data.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: [batch, 684, 20, 100]")
        print(f"Shapes match: {output.shape == (batch_size, 684, 20, 100)}")