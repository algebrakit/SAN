import os
import time
import argparse
import random
import torch
import numpy as np
from tensorboardX import SummaryWriter

from utils import load_config, save_checkpoint, load_checkpoint
from dataset_stroke import get_stroke_dataset
from models.Backbone_stroke import StrokeBackbone
from training_stroke import train_stroke, eval_stroke


def main():
    parser = argparse.ArgumentParser(description='SAN Stroke Training')
    parser.add_argument('--config', default='config_stroke.yaml', type=str, help='path to stroke config file')
    parser.add_argument('--check', action='store_true', help='only for code check')
    parser.add_argument('--no-cache', action='store_true', help='disable caching (force re-parse all files)')
    parser.add_argument('--rebuild-cache', action='store_true', help='rebuild cache even if it exists')
    parser.add_argument('--cache-only', action='store_true', help='only build cache and exit (useful for preprocessing)')
    args = parser.parse_args()

    if not args.config:
        print('Please provide stroke config yaml file')
        exit(-1)

    # Load configuration
    params = load_config(args.config)
    print(f"Loaded stroke configuration: {args.config}")
    print(f"Experiment: {params['experiment']}")
    
    # Override cache settings from command line
    if args.no_cache:
        params['use_cache'] = False
        print("Cache disabled via --no-cache")
    if args.rebuild_cache:
        params['force_rebuild_cache'] = True
        print("Cache rebuild forced via --rebuild-cache")

    # Set random seed
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    print(f"Using device: {device}")

    # Create data loaders for stroke data
    print("Loading stroke datasets...")
    train_loader, eval_loader = get_stroke_dataset(params)
    
    # If cache-only mode, exit after building cache
    if args.cache_only:
        print("✓ Cache building completed. Use --check or start training normally.")
        return

    # Create stroke-aware model
    print("Creating stroke-aware SAN model...")
    model = StrokeBackbone(params)
    
    # Generate model name with timestamp
    now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    model.name = f'{params["experiment"]}_{now}_StrokeEncoder_Decoder-{params["decoder"]["net"]}_' \
                 f'max_strokes-{params["max_strokes"]}_max_points-{params["max_points_per_stroke"]}'
    print(f"Model name: {model.name}")
    
    model = model.to(device)

    # Create TensorBoard writer
    if args.check:
        writer = None
        print("Code check mode - no logging")
    else:
        os.makedirs(params["log_dir"], exist_ok=True)
        writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')
        print(f"TensorBoard logging to: {params['log_dir']}/{model.name}")

    # Create optimizer
    optimizer = getattr(torch.optim, params['optimizer'])(
        model.parameters(), 
        lr=float(params['lr']),
        eps=float(params['eps']), 
        weight_decay=float(params['weight_decay'])
    )

    # Create checkpoint directory
    os.makedirs(params['checkpoint_dir'], exist_ok=True)

    # Load checkpoint if specified
    start_epoch = 0
    if params['checkpoint']:
        print(f"Loading checkpoint: {params['checkpoint']}")
        load_checkpoint(model, optimizer, params['checkpoint'])
        start_epoch = 0  # Will be updated to resume from correct epoch
        print(f"Resumed from epoch {start_epoch}")

    print("\n" + "="*80)
    print("STROKE-AWARE SAN TRAINING")
    print("="*80)
    print(f"Dataset: {len(train_loader.dataset)} training samples, {len(eval_loader.dataset)} eval samples")
    print(f"Batch size: {params['batch_size']}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Evaluation batches: {len(eval_loader)}")
    print(f"Max strokes per expression: {params['max_strokes']}")
    print(f"Max points per stroke: {params['max_points_per_stroke']}")
    print(f"Stroke coordinate range: width={params['stroke_target_width']}, height={params['stroke_target_height']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("="*80)

    if args.check:
        print("+ Code check passed - all components loaded successfully")
        print("+ Model architecture:")
        print(f"  - Point LSTM: {params['point_lstm_hidden']} hidden units, {params['point_lstm_layers']} layers")
        print(f"  - Stroke Transformer: {params['transformer_heads']} heads, {params['transformer_layers']} layers")
        print(f"  - Output feature maps: {params['encoder']['out_channels']} channels, {params['feature_height']}x{params['feature_width']}")
        
        # Test with a small batch
        print("+ Testing forward pass...")
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= 1:  # Only test one batch
                break
            
            try:
                stroke_data, stroke_masks, stroke_positions, labels, labels_mask = batch_data
                stroke_data = stroke_data.to(device)
                stroke_masks = stroke_masks.to(device)  
                stroke_positions = stroke_positions.to(device)
                labels = labels.to(device)
                labels_mask = labels_mask.to(device)
                
                with torch.no_grad():
                    predictions, losses = model(stroke_data, stroke_masks, stroke_positions, labels, labels_mask)
                    print(f"+ Forward pass successful")
                    print(f"  - Input shape: {stroke_data.shape}")
                    print(f"  - Word predictions shape: {predictions[0].shape}")
                    print(f"  - Struct predictions shape: {predictions[1].shape}")
                    print(f"  - Losses: {len(losses)} loss terms")
            except Exception as e:
                print(f"- Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
        
        return

    # Training loop
    best_eval_loss = float('inf')
    
    for epoch in range(start_epoch, params['epoches']):
        print(f"\nEpoch {epoch + 1}/{params['epoches']}")
        print("-" * 50)
        
        # Training
        train_loss, train_metrics = train_stroke(params, model, optimizer, epoch, train_loader, writer)
        
        # Evaluation
        eval_loss, eval_metrics = eval_stroke(params, model, epoch, eval_loader, writer)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.6f} | Eval Loss: {eval_loss:.6f}")
        if train_metrics:
            print(f"Train - Word: {train_metrics.get('word_acc', 0):.3f}, "
                  f"Struct: {train_metrics.get('struct_acc', 0):.3f}, "
                  f"Expr: {train_metrics.get('expr_acc', 0):.3f}")
        if eval_metrics:
            print(f"Eval  - Word: {eval_metrics.get('word_acc', 0):.3f}, "
                  f"Struct: {eval_metrics.get('struct_acc', 0):.3f}, "
                  f"Expr: {eval_metrics.get('expr_acc', 0):.3f}")

        # Save checkpoint
        checkpoint_path = f"{params['checkpoint_dir']}/{model.name}"
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save latest checkpoint
        save_checkpoint(model, optimizer, train_metrics.get('word_acc', 0), 
                       train_metrics.get('struct_acc', 0), train_metrics.get('expr_acc', 0),
                       epoch + 1, path=checkpoint_path)
        
        # Save best checkpoint
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_checkpoint(model, optimizer, eval_metrics.get('word_acc', 0),
                           eval_metrics.get('struct_acc', 0), eval_metrics.get('expr_acc', 0),
                           epoch + 1, path=checkpoint_path)
            print(f"+ New best model saved (eval loss: {eval_loss:.6f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, train_metrics.get('word_acc', 0),
                           train_metrics.get('struct_acc', 0), train_metrics.get('expr_acc', 0),
                           epoch + 1, path=checkpoint_path)

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best evaluation loss: {best_eval_loss:.6f}")
    print(f"Model checkpoints saved to: {checkpoint_path}")
    print(f"TensorBoard logs: {params['log_dir']}/{model.name}")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()