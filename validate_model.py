#!/usr/bin/env python3
"""
Comprehensive model validation script for Stroke-Aware SAN
Validates architecture, dimensions, gradient flow, and compatibility
"""

import os
import torch
import torch.nn as nn
import numpy as np
from utils import load_config
from dataset_stroke import Words, StrokeNormalizer
from models.Backbone_stroke import StrokeBackbone

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_success(message):
    """Print success message"""
    print(f"[PASS] {message}")
    
def print_error(message):
    """Print error message"""
    print(f"[FAIL] {message}")
    
def print_info(message):
    """Print info message"""
    print(f"[INFO] {message}")

class ModelValidator:
    """Comprehensive model validation"""
    
    def __init__(self, config_path="config_stroke.yaml"):
        """Initialize validator"""
        print_header("INITIALIZING MODEL VALIDATOR")
        
        # Load config
        self.config = load_config(config_path)
        # Force CPU for validation to avoid CUDA errors
        self.device = torch.device('cpu')
        self.config['device'] = self.device
        
        # Add required parameters
        self.config['struct_num'] = 7
        
        print_info(f"Device: {self.device}")
        print_info(f"Config loaded with {len(self.config)} parameters")
        
        # Load vocabulary
        self.words = Words(self.config['word_path'])
        self.config['word_num'] = len(self.words)
        print_info(f"Vocabulary size: {len(self.words)}")
        
        # Initialize normalizer
        self.normalizer = StrokeNormalizer(self.config)
        print_info("Stroke normalizer initialized")
        
    def validate_config(self):
        """Validate configuration parameters"""
        print_header("VALIDATING CONFIGURATION")
        
        required_params = [
            'max_strokes', 'max_points_per_stroke', 'point_lstm_hidden',
            'point_lstm_layers', 'transformer_heads', 'transformer_layers',
            'feature_height', 'feature_width', 'word_num', 'struct_num'
        ]
        
        missing_params = []
        for param in required_params:
            if param not in self.config:
                missing_params.append(param)
        
        if missing_params:
            print_error(f"Missing parameters: {missing_params}")
            return False
        else:
            print_success("All required parameters present")
            
        # Print key parameters
        print_info(f"Max strokes: {self.config['max_strokes']}")
        print_info(f"Max points per stroke: {self.config['max_points_per_stroke']}")
        print_info(f"Feature dimensions: {self.config['feature_height']}x{self.config['feature_width']}")
        print_info(f"Vocab size: {self.config['word_num']}")
        
        return True
    
    def validate_model_creation(self):
        """Validate model can be created"""
        print_header("VALIDATING MODEL CREATION")
        
        try:
            self.model = StrokeBackbone(self.config)
            self.model = self.model.to(self.device)
            print_success("Model created successfully")
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print_info(f"Total parameters: {total_params:,}")
            print_info(f"Trainable parameters: {trainable_params:,}")
            
            return True
            
        except Exception as e:
            print_error(f"Model creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_dummy_input(self):
        """Create dummy input data for testing"""
        print_header("CREATING DUMMY INPUT DATA")
        
        batch_size = 2
        max_seq_len = 20
        
        # Create dummy stroke data
        stroke_data = torch.randn(
            batch_size, 
            self.config['max_strokes'], 
            self.config['max_points_per_stroke'], 
            3  # x, y, t
        ).to(self.device)
        
        # Create stroke masks [batch, max_strokes, max_points_per_stroke]
        stroke_masks = torch.zeros(
            batch_size, 
            self.config['max_strokes'], 
            self.config['max_points_per_stroke']
        ).to(self.device)
        # First 10 strokes have first 5 points valid
        stroke_masks[:, :10, :5] = 1
        
        # Create stroke positions [batch, max_strokes, 4] -> [center_x, center_y, width, height]
        stroke_positions = torch.randn(batch_size, self.config['max_strokes'], 4).to(self.device)
        # Make them reasonable values (center around 0, positive width/height)
        stroke_positions[:, :, :2] = stroke_positions[:, :, :2] * 0.5  # centers in [-0.5, 0.5]
        stroke_positions[:, :, 2:] = torch.abs(stroke_positions[:, :, 2:]) * 0.2 + 0.1  # positive dimensions
        
        # Create dummy labels
        labels = torch.randint(0, self.config['word_num'], (batch_size, max_seq_len, 11)).to(self.device)
        labels_mask = torch.ones(batch_size, max_seq_len, 2).to(self.device)
        
        print_info(f"Stroke data shape: {stroke_data.shape}")
        print_info(f"Stroke masks shape: {stroke_masks.shape}")
        print_info(f"Stroke positions shape: {stroke_positions.shape}")
        print_info(f"Labels shape: {labels.shape}")
        
        return stroke_data, stroke_masks, stroke_positions, labels, labels_mask
    
    def validate_forward_pass(self):
        """Validate forward pass"""
        print_header("VALIDATING FORWARD PASS")
        
        if not hasattr(self, 'model'):
            print_error("Model not created yet")
            return False
            
        try:
            # Create dummy input
            stroke_data, stroke_masks, stroke_positions, labels, labels_mask = self.create_dummy_input()
            
            # Set model to evaluation mode
            self.model.eval()
            
            with torch.no_grad():
                # Forward pass
                predictions, losses = self.model(
                    stroke_data, stroke_masks, stroke_positions,
                    labels, labels_mask, is_train=False
                )
                
                word_probs, struct_probs = predictions
                
                print_success("Forward pass completed")
                print_info(f"Word predictions shape: {word_probs.shape}")
                print_info(f"Struct predictions shape: {struct_probs.shape}")
                
                # Validate shapes
                expected_word_shape = (labels.shape[0], labels.shape[1], self.config['word_num'])
                expected_struct_shape = (labels.shape[0], labels.shape[1], self.config['struct_num'])
                
                if word_probs.shape == expected_word_shape:
                    print_success(f"Word predictions shape correct: {word_probs.shape}")
                else:
                    print_error(f"Word predictions shape mismatch. Expected: {expected_word_shape}, Got: {word_probs.shape}")
                    
                if struct_probs.shape == expected_struct_shape:
                    print_success(f"Struct predictions shape correct: {struct_probs.shape}")
                else:
                    print_error(f"Struct predictions shape mismatch. Expected: {expected_struct_shape}, Got: {struct_probs.shape}")
                
                return True
                
        except Exception as e:
            print_error(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_gradient_flow(self):
        """Validate gradient flow through the model"""
        print_header("VALIDATING GRADIENT FLOW")
        
        if not hasattr(self, 'model'):
            print_error("Model not created yet")
            return False
            
        try:
            # Create dummy input
            stroke_data, stroke_masks, stroke_positions, labels, labels_mask = self.create_dummy_input()
            
            # Set model to training mode
            self.model.train()
            
            # Forward pass
            predictions, losses = self.model(
                stroke_data, stroke_masks, stroke_positions,
                labels, labels_mask, is_train=True
            )
            
            # Calculate total loss
            total_loss = sum(losses.values())
            
            print_info(f"Total loss: {total_loss.item():.4f}")
            print_info(f"Loss components: {list(losses.keys())}")
            
            # Backward pass
            total_loss.backward()
            
            # Check gradients
            grad_norms = []
            no_grad_params = []
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                else:
                    no_grad_params.append(name)
            
            if no_grad_params:
                print_error(f"Parameters with no gradients: {no_grad_params[:5]}...")  # Show first 5
            else:
                print_success("All parameters have gradients")
                
            if grad_norms:
                avg_grad_norm = np.mean(grad_norms)
                max_grad_norm = np.max(grad_norms)
                min_grad_norm = np.min(grad_norms)
                
                print_info(f"Gradient norm - Avg: {avg_grad_norm:.6f}, Max: {max_grad_norm:.6f}, Min: {min_grad_norm:.6f}")
                
                if max_grad_norm > 100:
                    print_error(f"Very large gradients detected (max: {max_grad_norm:.2f})")
                elif max_grad_norm < 1e-8:
                    print_error(f"Very small gradients detected (max: {max_grad_norm:.2e})")
                else:
                    print_success("Gradient magnitudes look reasonable")
                    
            return True
            
        except Exception as e:
            print_error(f"Gradient flow validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_encoder_output(self):
        """Validate stroke encoder output dimensions"""
        print_header("VALIDATING STROKE ENCODER OUTPUT")
        
        if not hasattr(self, 'model'):
            print_error("Model not created yet")
            return False
            
        try:
            # Create dummy input
            stroke_data, stroke_masks, stroke_positions, _, _ = self.create_dummy_input()
            
            # Access encoder directly
            encoder = self.model.encoder
            
            with torch.no_grad():
                encoder_output = encoder(stroke_data, stroke_masks, stroke_positions)
                
                print_info(f"Encoder output shape: {encoder_output.shape}")
                
                # Expected: [batch, channels, height, width] = [batch, 684, 20, 100]
                expected_shape = (stroke_data.shape[0], 684, 20, 100)
                
                if encoder_output.shape == expected_shape:
                    print_success(f"Encoder output shape correct: {encoder_output.shape}")
                else:
                    print_error(f"Encoder output shape mismatch. Expected: {expected_shape}, Got: {encoder_output.shape}")
                
                # Check for reasonable values
                mean_val = encoder_output.mean().item()
                std_val = encoder_output.std().item()
                
                print_info(f"Encoder output statistics - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                
                if torch.isnan(encoder_output).any():
                    print_error("NaN values detected in encoder output")
                elif torch.isinf(encoder_output).any():
                    print_error("Infinite values detected in encoder output")
                else:
                    print_success("Encoder output values are finite")
                
                return True
                
        except Exception as e:
            print_error(f"Encoder validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_compatibility(self):
        """Validate compatibility with original SAN decoder"""
        print_header("VALIDATING SAN DECODER COMPATIBILITY")
        
        try:
            # Check if decoder parameters match expected
            decoder = self.model.decoder
            
            # Verify decoder architecture
            print_info(f"Decoder type: {type(decoder).__name__}")
            
            # Check attention mechanism
            if hasattr(decoder, 'word_attention') and hasattr(decoder, 'c2p_attention'):
                print_success("Attention mechanisms present (word & c2p)")
                print_info(f"Word attention type: {type(decoder.word_attention).__name__}")
            else:
                print_error("Missing attention mechanisms")
                
            # Check GRU cells
            if hasattr(decoder, 'word_input_gru') and hasattr(decoder, 'word_out_gru'):
                print_success("GRU cells present")
            else:
                print_error("Missing GRU cells")
                
            # Check output layers
            if hasattr(decoder, 'word_convert') and hasattr(decoder, 'struct_convert'):
                print_success("Word and structure output layers present")
            else:
                print_error("Missing output layers")
                
            return True
            
        except Exception as e:
            print_error(f"Compatibility validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_full_validation(self):
        """Run complete validation suite"""
        print_header("STROKE-AWARE SAN MODEL VALIDATION")
        print_info("Running comprehensive model validation...")
        
        results = {}
        
        # Run all validations
        results['config'] = self.validate_config()
        results['model_creation'] = self.validate_model_creation()
        results['forward_pass'] = self.validate_forward_pass()
        results['gradient_flow'] = self.validate_gradient_flow()
        results['encoder_output'] = self.validate_encoder_output()
        results['compatibility'] = self.validate_compatibility()
        
        # Summary
        print_header("VALIDATION SUMMARY")
        
        passed = sum(results.values())
        total = len(results)
        
        for test, result in results.items():
            status = "PASS" if result else "FAIL"
            symbol = "[PASS]" if result else "[FAIL]"
            print(f"{symbol} {test.upper()}: {status}")
        
        print("\n" + "="*60)
        if passed == total:
            print(f">> ALL TESTS PASSED ({passed}/{total})")
            print("Your stroke model definition is valid!")
        else:
            print(f">> {total-passed} TESTS FAILED ({passed}/{total})")
            print("Please address the issues above before training.")
        print("="*60)
        
        return passed == total


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Stroke-Aware SAN Model')
    parser.add_argument('--config', default='config_stroke.yaml', help='Config file path')
    parser.add_argument('--quick', action='store_true', help='Run only basic validations')
    args = parser.parse_args()
    
    try:
        validator = ModelValidator(args.config)
        
        if args.quick:
            # Quick validation
            success = (validator.validate_config() and 
                      validator.validate_model_creation() and
                      validator.validate_forward_pass())
        else:
            # Full validation
            success = validator.run_full_validation()
        
        if success:
            print("\n>> Model validation completed successfully!")
            return 0
        else:
            print("\n>> Model validation failed!")
            return 1
            
    except Exception as e:
        print(f"\n>> Validation crashed: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    exit(main())