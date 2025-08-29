import os
import argparse
import torch
import numpy as np
from datetime import datetime

from utils import load_config, load_checkpoint
from dataset_stroke import StrokeNormalizer, Words
from models.Backbone_stroke import StrokeBackbone
from latex_parser import LaTeXToSANConverter


class StrokeInference:
    """Inference engine for stroke-aware SAN model"""
    
    def __init__(self, config_path, checkpoint_path):
        print(">> Initializing Stroke-Aware SAN Inference...")
        
        # Load configuration
        self.params = load_config(config_path)
        print(f">> Loaded config: {config_path}")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params['device'] = self.device
        print(f">> Using device: {self.device}")
        
        # Load vocabulary
        self.words = Words(self.params['word_path'])
        self.params['word_num'] = len(self.words)
        # Add missing parameters for decoder compatibility
        self.params['struct_num'] = 7  # Same as original SAN
        print(f">> Vocabulary loaded: {len(self.words)} symbols")
        
        # Initialize stroke normalizer
        self.normalizer = StrokeNormalizer(self.params)
        print(">> Stroke normalizer initialized")
        
        # Create model
        print(">> Building stroke-aware model...")
        self.model = StrokeBackbone(self.params)
        self.model = self.model.to(self.device)
        
        # Load checkpoint
        print(f">> Loading checkpoint: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            print(f">> Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            word_acc = checkpoint.get('word_acc', 'unknown')
            struct_acc = checkpoint.get('struct_acc', 'unknown')
            if isinstance(word_acc, (int, float)):
                print(f"   Word accuracy: {word_acc:.4f}")
            else:
                print(f"   Word accuracy: {word_acc}")
            if isinstance(struct_acc, (int, float)):
                print(f"   Struct accuracy: {struct_acc:.4f}")
            else:
                print(f"   Struct accuracy: {struct_acc}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Set to evaluation mode
        self.model.eval()
        print(">> Model ready for inference!")
        print("=" * 60)
        
    def predict_from_inkml(self, inkml_path):
        """Make prediction from InkML file"""
        print(f"\n>> Processing InkML: {os.path.basename(inkml_path)}")
        
        # Parse InkML file
        parsed_data = self.normalizer.parse_inkml(inkml_path)
        if parsed_data is None:
            print(">> Failed to parse InkML file")
            return None
            
        print(f"   Original label: {parsed_data['label']}")
        print(f"   Number of strokes: {len(parsed_data['strokes'])}")
        
        # Normalize strokes
        normalized_data = self.normalizer.normalize_expression(parsed_data['strokes'])
        
        # Convert to tensor format
        stroke_tensor = torch.zeros(self.normalizer.max_strokes, 
                                   self.normalizer.max_points_per_stroke, 3)
        
        for stroke_idx, stroke in enumerate(normalized_data['normalized_strokes']):
            for point_idx, (x, y, t) in enumerate(stroke['points']):
                stroke_tensor[stroke_idx, point_idx, 0] = x
                stroke_tensor[stroke_idx, point_idx, 1] = y  
                stroke_tensor[stroke_idx, point_idx, 2] = t
        
        # Add batch dimension and move to device
        stroke_tensor = stroke_tensor.unsqueeze(0).to(self.device)
        stroke_masks = normalized_data['stroke_masks'].unsqueeze(0).to(self.device)
        stroke_positions = normalized_data['stroke_positions'].unsqueeze(0).to(self.device)
        
        print(f"   Input tensor shape: {stroke_tensor.shape}")
        print(f"   Valid strokes: {normalized_data['num_valid_strokes']}")
        
        # Run inference
        with torch.no_grad():
            # Create dummy labels for inference (not used in prediction)
            batch_size = 1
            max_length = self.params.get('max_sequence_length', 50)
            dummy_labels = torch.zeros(batch_size, max_length, 11, dtype=torch.long).to(self.device)
            dummy_labels_mask = torch.zeros(batch_size, max_length, 2).to(self.device)
            
            # Forward pass
            predictions, losses = self.model(
                stroke_tensor, stroke_masks, stroke_positions, 
                dummy_labels, dummy_labels_mask, is_train=False
            )
            
            word_probs, struct_probs = predictions
            
            # Decode predictions
            predicted_sequence = self.decode_predictions(word_probs, struct_probs)
            
        print(f">> Prediction complete!")
        return {
            'original_label': parsed_data['label'],
            'predicted_latex': predicted_sequence,
            'confidence_scores': word_probs[0].max(dim=-1)[0].cpu().numpy(),
            'num_valid_strokes': normalized_data['num_valid_strokes'],
            'normalization_info': normalized_data['expression_metadata']
        }
    
    def decode_predictions(self, word_probs, struct_probs):
        """Decode model predictions to LaTeX string"""
        # Get most likely word sequence
        word_predictions = word_probs[0].argmax(dim=-1)  # [seq_length]
        
        # Convert to words
        predicted_words = []
<<<<<<< Updated upstream
        for token_id in word_predictions:
            token_id = token_id.item()
            if token_id == 0:  # End of sequence
                break
            if token_id < len(self.words.words_index_dict):
                word = self.words.words_index_dict[token_id]
                if word != '<pad>':  # Skip padding tokens
                    predicted_words.append(word)
        
        # Join words to form LaTeX expression
=======
        for i, token_id in enumerate(word_predictions):
            token_id = token_id.item()
            
            # Check for end-of-sequence tokens
            if token_id == 0 or (token_id < len(self.words.words_index_dict) and 
                                self.words.words_index_dict[token_id] == '<eos>'):
                break
                
            # Validate token ID bounds
            if token_id >= len(self.words.words_index_dict):
                print(f"Warning: Invalid token ID {token_id} at position {i}, max vocab size: {len(self.words.words_index_dict)}")
                continue
                
            word = self.words.words_index_dict[token_id]
            
            # Skip special tokens
            if word not in ['<pad>', '<sos>', '<eos>', 'struct']:
                predicted_words.append(word)
        
        # Join words to form LaTeX expression, handle empty case
        if not predicted_words:
            return "<empty_prediction>"
            
>>>>>>> Stashed changes
        latex_expression = ' '.join(predicted_words)
        return latex_expression
    
    def predict_from_strokes(self, strokes_data):
        """Make prediction from raw stroke data (for programmatic input)"""
        print(f"\n>> Processing {len(strokes_data)} stroke(s)...")
        
        # Convert raw strokes to normalized format
        formatted_strokes = []
        for i, stroke in enumerate(strokes_data):
            formatted_strokes.append({
                'id': f'stroke_{i}',
                'points': stroke  # Expected: list of (x, y, t) tuples
            })
        
        # Normalize strokes
        normalized_data = self.normalizer.normalize_expression(formatted_strokes)
        
        # Convert to tensor format (same as InkML processing)
        stroke_tensor = torch.zeros(self.normalizer.max_strokes, 
                                   self.normalizer.max_points_per_stroke, 3)
        
        for stroke_idx, stroke in enumerate(normalized_data['normalized_strokes']):
            for point_idx, (x, y, t) in enumerate(stroke['points']):
                stroke_tensor[stroke_idx, point_idx, 0] = x
                stroke_tensor[stroke_idx, point_idx, 1] = y  
                stroke_tensor[stroke_idx, point_idx, 2] = t
        
        # Add batch dimension and move to device
        stroke_tensor = stroke_tensor.unsqueeze(0).to(self.device)
        stroke_masks = normalized_data['stroke_masks'].unsqueeze(0).to(self.device)
        stroke_positions = normalized_data['stroke_positions'].unsqueeze(0).to(self.device)
        
        # Run inference (same as InkML)
        with torch.no_grad():
            batch_size = 1
            max_length = self.params.get('max_sequence_length', 50)
            dummy_labels = torch.zeros(batch_size, max_length, 11, dtype=torch.long).to(self.device)
            dummy_labels_mask = torch.zeros(batch_size, max_length, 2).to(self.device)
            
            predictions, losses = self.model(
                stroke_tensor, stroke_masks, stroke_positions, 
                dummy_labels, dummy_labels_mask, is_train=False
            )
            
            word_probs, struct_probs = predictions
            predicted_sequence = self.decode_predictions(word_probs, struct_probs)
            
        return {
            'predicted_latex': predicted_sequence,
            'confidence_scores': word_probs[0].max(dim=-1)[0].cpu().numpy(),
            'num_valid_strokes': normalized_data['num_valid_strokes']
        }


def main():
    parser = argparse.ArgumentParser(description='Stroke-Aware SAN Inference')
    parser.add_argument('--config', default='config_stroke.yaml', 
                       help='Path to stroke config file')
    parser.add_argument('--checkpoint', required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--inkml', help='Path to InkML file for prediction')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with programmatic stroke input')
    args = parser.parse_args()
    
    if not args.config or not args.checkpoint:
        print(">> Please provide both config and checkpoint paths")
        return
        
    # Initialize inference engine
    try:
        inference = StrokeInference(args.config, args.checkpoint)
    except Exception as e:
        print(f">> Failed to initialize inference: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run inference based on input type
    if args.inkml:
        if not os.path.exists(args.inkml):
            print(f">> InkML file not found: {args.inkml}")
            return
            
        result = inference.predict_from_inkml(args.inkml)
        if result:
            print("\n" + "=" * 60)
            print(">> INFERENCE RESULTS")
            print("=" * 60)
            print(f"Original label: {result['original_label']}")
            print(f"Predicted LaTeX: {result['predicted_latex']}")
            print(f"Valid strokes: {result['num_valid_strokes']}")
            print(f"Confidence: {result['confidence_scores'][:10]}")  # First 10 tokens
            
    elif args.demo:
        print("\n>> Running demo with programmatic stroke input...")
        
        # Create simple demo strokes (e.g., drawing a "2")
        demo_strokes = [
            # Stroke 1: Top horizontal line
            [(0.0, 0.8, 0.0), (0.5, 0.8, 0.1), (1.0, 0.8, 0.2)],
            # Stroke 2: Diagonal line
            [(1.0, 0.8, 0.3), (0.0, 0.0, 0.4)],
            # Stroke 3: Bottom horizontal line
            [(0.0, 0.0, 0.5), (0.5, 0.0, 0.6), (1.0, 0.0, 0.7)]
        ]
        
        result = inference.predict_from_strokes(demo_strokes)
        print("\n" + "=" * 60)
        print(">> DEMO RESULTS")
        print("=" * 60)
        print(f"Predicted LaTeX: {result['predicted_latex']}")
        print(f"Valid strokes: {result['num_valid_strokes']}")
        print(f"Confidence: {result['confidence_scores'][:10]}")
        
    else:
        print(">> Please provide either --inkml path or --demo flag")
        print("\nUsage examples:")
        print(f"  python {__file__} --config config_stroke.yaml --checkpoint your_model.pth --inkml sample.inkml")
        print(f"  python {__file__} --config config_stroke.yaml --checkpoint your_model.pth --demo")


if __name__ == '__main__':
    main()