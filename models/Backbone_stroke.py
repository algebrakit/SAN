import torch.nn as nn
import torch
from models.stroke_encoder import create_stroke_encoder
import models
from models import SAN_decoder  # Import the existing decoder


class StrokeBackbone(nn.Module):
    """Stroke-aware Backbone that replaces CNN encoder with stroke processing"""
    
    def __init__(self, params=None):
        super(StrokeBackbone, self).__init__()

        self.params = params
        self.use_label_mask = params['use_label_mask']

        # Create stroke-aware encoder instead of CNN
        self.encoder = create_stroke_encoder(params)
        
        # Use existing SAN decoder (unchanged)
        self.decoder = getattr(models, params['decoder']['net'])(params=self.params)
        
        # Loss functions (same as original)
        self.cross = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss(reduction='none')
        
        # Ratio for spatial feature map dimensions (keep for compatibility)  
        # Use same ratio as DenseNet for proper mask downsampling in KL loss
        self.ratio = 16  # Match original DenseNet ratio

    def forward(self, stroke_data, stroke_masks, stroke_positions, labels, labels_mask, is_train=True):
        """
        Forward pass with stroke data instead of images
        
        Args:
            stroke_data: [batch, max_strokes, max_points, 3] - normalized stroke coordinates
            stroke_masks: [batch, max_strokes, max_points] - point-level masks  
            stroke_positions: [batch, max_strokes, 4] - stroke spatial positions
            labels: [batch, max_length, 11] - ground truth labels
            labels_mask: [batch, max_length, 2] - label masks
            is_train: bool - training mode flag
            
        Returns:
            If training: (predictions, losses)
            If inference: (predictions, losses)  
        """
        
        # Generate spatial features from strokes
        cnn_features = self.encoder(stroke_data, stroke_masks, stroke_positions)
        
        # Create image mask for decoder compatibility
        # The decoder expects image_mask with shape [batch, 1, height, width]
        batch_size = stroke_data.shape[0]
        feature_height, feature_width = cnn_features.shape[2], cnn_features.shape[3]
        
        # Generate image mask based on valid strokes
        # Simple approach: if any stroke exists in spatial region, mark as valid
        stroke_valid_mask = stroke_masks.sum(dim=2) > 0  # [batch, max_strokes] - strokes with points
        
        # Create spatial mask - for now, mark entire feature map as valid if any strokes exist
        images_mask = torch.ones(batch_size, 1, feature_height, feature_width, 
                               device=cnn_features.device, dtype=torch.float32)
        
        # If no valid strokes in batch, mask everything
        has_valid_strokes = stroke_valid_mask.sum(dim=1) > 0  # [batch]
        for batch_idx in range(batch_size):
            if not has_valid_strokes[batch_idx]:
                images_mask[batch_idx] = 0.0
        
        # Pass through decoder (same interface as original)
        word_probs, struct_probs, words_alphas, struct_alphas, c2p_probs, c2p_alphas = self.decoder(
            cnn_features, labels, images_mask, labels_mask, is_train=is_train
        )

        # Compute losses (same as original Backbone)
        word_average_loss = self.cross(
            word_probs.contiguous().view(-1, word_probs.shape[-1]), 
<<<<<<< Updated upstream
            labels[:, :, 1].view(-1)
=======
            labels[:, :, 0].view(-1)
>>>>>>> Stashed changes
        )

        struct_probs = torch.sigmoid(struct_probs)
        struct_average_loss = self.bce(struct_probs, labels[:, :, 4:].float())
        
        if labels_mask is not None:
            struct_average_loss = (struct_average_loss * labels_mask[:, :, 0][:, :, None]).sum() / (labels_mask[:, :, 0].sum() + 1e-10)

        if is_train:
            parent_average_loss = self.cross(
                c2p_probs.contiguous().view(-1, word_probs.shape[-1]), 
                labels[:, :, 3].view(-1)
            )
            
            kl_average_loss = self.cal_kl_loss(
                words_alphas, c2p_alphas, labels, 
                images_mask[:, :, ::self.ratio, ::self.ratio], labels_mask
            )

            return (word_probs, struct_probs), (word_average_loss, struct_average_loss, parent_average_loss, kl_average_loss)

        return (word_probs, struct_probs), (word_average_loss, struct_average_loss)

    def cal_kl_loss(self, child_alphas, parent_alphas, labels, image_mask, label_mask):
        """KL divergence loss for attention consistency (same as original)"""
        batch_size, steps, height, width = child_alphas.shape
        
        # Downsample alpha tensors to match downsampled image_mask
        # image_mask is downsampled by self.ratio, so downsample alphas too
        child_alphas_down = child_alphas[:, :, ::self.ratio, ::self.ratio]
        parent_alphas_down = parent_alphas[:, :, ::self.ratio, ::self.ratio]
        
        _, _, down_height, down_width = child_alphas_down.shape
        
        new_child_alphas = torch.zeros((batch_size, steps, down_height, down_width)).to(self.params['device'])
        new_child_alphas[:, 1:, :, :] = child_alphas_down[:, :-1, :, :].clone()
        new_child_alphas = new_child_alphas.view((batch_size * steps, down_height, down_width))
        parent_ids = labels[:, :, 2] + steps * torch.arange(batch_size)[:, None].to(self.params['device'])

        new_child_alphas = new_child_alphas[parent_ids]
        new_child_alphas = new_child_alphas.view((batch_size, steps, down_height, down_width))[:, 1:, :, :]
        new_parent_alphas = parent_alphas_down[:, 1:, :, :]

        KL_alpha = new_child_alphas * (torch.log(new_child_alphas + 1e-10) - torch.log(new_parent_alphas + 1e-10)) * image_mask
        KL_loss = (KL_alpha.sum(-1).sum(-1) * label_mask[:, :-1, 0]).sum(-1).sum(-1) / (label_mask.sum() - batch_size)

        return KL_loss


# Helper function to create stroke backbone
def create_stroke_backbone(params):
    """Factory function for creating stroke-aware backbone"""
    return StrokeBackbone(params)


if __name__ == '__main__':
    # Test stroke backbone
    params = {
        'encoder': {'out_channels': 684},
        'decoder': {'net': 'SAN_decoder', 'hidden_size': 256, 'input_size': 256, 'cell': 'GRU'},
        'attention': {'attention_dim': 512},
        'word_num': 1000,
        'struct_num': 7,
        'use_label_mask': True,
        'device': 'cpu',
        'max_strokes': 20,
        'max_points_per_stroke': 100,
        'feature_height': 20,
        'feature_width': 100,
        'point_lstm_hidden': 128,
        'transformer_heads': 8,
        'transformer_layers': 3
    }
    
    # Create test data
    batch_size = 2
    max_strokes = 20
    max_points = 100
    max_length = 15
    
    stroke_data = torch.randn(batch_size, max_strokes, max_points, 3)
    stroke_masks = torch.randint(0, 2, (batch_size, max_strokes, max_points)).float()
    stroke_positions = torch.randn(batch_size, max_strokes, 4)
    labels = torch.randint(0, 100, (batch_size, max_length, 11)).long()
    labels_mask = torch.randint(0, 2, (batch_size, max_length, 2)).float()
    
    # Test backbone creation
    print("Creating stroke backbone...")
    try:
        backbone = StrokeBackbone(params)
        print("✓ Stroke backbone created successfully")
        
        # Test forward pass shape compatibility
        with torch.no_grad():
            encoder_output = backbone.encoder(stroke_data, stroke_masks, stroke_positions)
            print(f"✓ Encoder output shape: {encoder_output.shape}")
            print(f"✓ Expected: [batch_size, 684, 20, 100] = [{batch_size}, 684, 20, 100]")
            print(f"✓ Shapes match: {encoder_output.shape == (batch_size, 684, 20, 100)}")
        
    except Exception as e:
        print(f"✗ Error creating stroke backbone: {e}")
        import traceback
        traceback.print_exc()