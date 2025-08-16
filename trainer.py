"""
Training Module for Stroke Transformer

This module handles model training, evaluation, and progress tracking
with comprehensive metrics and visualization support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, top_k_accuracy_score
)
from pathlib import Path
import time
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from transformer_model import StrokeTransformer
from dataset_loader import DataManager


class EarlyStopping:
    """Early stopping to avoid overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.best_state_dict = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_state_dict = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            if self.restore_best and self.best_state_dict:
                model.load_state_dict(self.best_state_dict)
        
        return self.should_stop


class MetricsTracker:
    """Track and store training metrics."""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_current_metrics(self) -> Dict:
        return {key: values[-1] if values else 0 for key, values in self.metrics.items()}
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, path: Path):
        with open(path, 'r') as f:
            self.metrics = json.load(f)


class StrokeTransformerTrainer:
    """Complete training pipeline for StrokeTransformer."""
    
    def __init__(self,
                 model: StrokeTransformer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 class_weights: Optional[torch.Tensor] = None,
                 device: str = 'auto',
                 save_dir: Path = Path('checkpoints')):
        """
        Initialize trainer.
        
        Args:
            model: StrokeTransformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            class_weights: Optional class weights for imbalanced dataset
            device: Device to use ('auto', 'cuda', 'cpu')
            save_dir: Directory to save checkpoints and results
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
    
    def configure_optimizer(self,
                           optimizer_type: str = 'adamw',
                           learning_rate: float = 1e-4,
                           weight_decay: float = 0.01,
                           **optimizer_kwargs) -> optim.Optimizer:
        """Configure optimizer."""
        if optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
                **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        return self.optimizer
    
    def configure_scheduler(self,
                           scheduler_type: str = 'cosine',
                           **scheduler_kwargs):
        """Configure learning rate scheduler."""
        if scheduler_type.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_kwargs.get('T_max', 100),
                eta_min=scheduler_kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type.lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_kwargs.get('step_size', 30),
                gamma=scheduler_kwargs.get('gamma', 0.1)
            )
        elif scheduler_type.lower() == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_kwargs.get('factor', 0.5),
                patience=scheduler_kwargs.get('patience', 10),
                min_lr=scheduler_kwargs.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        
        return self.scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (features, labels) in enumerate(progress_bar):
            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.global_step += 1
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Update progress bar
            current_acc = correct_predictions / total_predictions
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct_predictions / total_predictions
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
    
    def evaluate(self, data_loader: DataLoader, desc: str = "Evaluating") -> Dict[str, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=desc)
            for features, labels in progress_bar:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def compute_detailed_metrics(self, predictions: List[int], labels: List[int], 
                               class_names: Optional[List[str]] = None) -> Dict:
        """Compute detailed evaluation metrics."""
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Top-k accuracy (if applicable)
        # Note: This requires predicted probabilities, which we don't have here
        # top_k_acc = top_k_accuracy_score(labels, predictions, k=5)
        
        metrics = {
            'accuracy': accuracy,
            'weighted_precision': precision,
            'weighted_recall': recall,
            'weighted_f1': f1,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'per_class_f1': per_class_f1,
            'per_class_support': per_class_support,
            'confusion_matrix': cm,
            'classification_report': classification_report(
                labels, predictions, 
                target_names=class_names, 
                zero_division=0
            )
        }
        
        return metrics
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            save_path: Optional[Path] = None, figsize: Tuple[int, int] = (12, 10)):
        """Plot confusion matrix."""
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        
        # Create heatmap
        sns.heatmap(cm_normalized, 
                   annot=False,  # Don't annotate due to potentially large number of classes
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_training_curves(self, save_path: Optional[Path] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.metrics_tracker.metrics['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.metrics_tracker.metrics['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.metrics_tracker.metrics['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.metrics_tracker.metrics['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.metrics_tracker.metrics['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        if self.metrics_tracker.metrics['learning_rate']:
            axes[1, 0].plot(epochs, self.metrics_tracker.metrics['learning_rate'], 'g-')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Epoch time
        if self.metrics_tracker.metrics['epoch_time']:
            axes[1, 1].plot(epochs, self.metrics_tracker.metrics['epoch_time'], 'm-')
            axes[1, 1].set_title('Epoch Training Time')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_acc': self.best_val_acc
        }
        
        if hasattr(self, 'scheduler') and self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            print(f"New best model saved with validation accuracy: {metrics['val_accuracy']:.4f}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self,
             epochs: int = 100,
             early_stopping_patience: int = 15,
             save_every: int = 10,
             evaluate_every: int = 1) -> Dict:
        """
        Main training loop.
        
        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_every: Save checkpoint every N epochs
            evaluate_every: Evaluate on validation set every N epochs
            
        Returns:
            Training history and final metrics
        """
        print("Starting training...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters")
        
        # Early stopping - DISABLED
        # early_stopping = EarlyStopping(patience=early_stopping_patience)
        print("Early stopping completely disabled - will train for all epochs")
        
        # Training loop
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            if epoch % evaluate_every == 0:
                val_metrics = self.evaluate(self.val_loader, desc=f"Validation Epoch {epoch}")
            else:
                val_metrics = {'loss': float('inf'), 'accuracy': 0.0}
            
            epoch_time = time.time() - epoch_start_time
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if hasattr(self, 'scheduler') and self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Track metrics
            self.metrics_tracker.update(
                train_loss=train_metrics['loss'],
                train_acc=train_metrics['accuracy'],
                val_loss=val_metrics['loss'],
                val_acc=val_metrics['accuracy'],
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            
            # Print epoch summary
            print(f"Epoch {epoch:3d}/{epochs}: "
                  f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.2f}s")
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
            
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, {
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                }, is_best=is_best)
            
            # Early stopping - DISABLED
            # if early_stopping(val_metrics['loss'], self.model):
            #     print(f"Early stopping at epoch {epoch}")
            #     break
        
        # Save final metrics
        self.metrics_tracker.save(self.save_dir / 'training_metrics.json')
        
        # Final evaluation
        print("\nEvaluating final model...")
        final_metrics = self.final_evaluation()
        
        return {
            'training_history': self.metrics_tracker.metrics,
            'final_metrics': final_metrics,
            'best_val_accuracy': self.best_val_acc,
            'total_epochs': self.epoch
        }
    
    def final_evaluation(self, class_names: Optional[List[str]] = None) -> Dict:
        """Perform comprehensive final evaluation."""
        # Load best model if available
        if self.best_model_path and self.best_model_path.exists():
            self.load_checkpoint(self.best_model_path)
        
        # Evaluate on all sets
        train_results = self.evaluate(self.train_loader, "Final Train Evaluation")
        val_results = self.evaluate(self.val_loader, "Final Validation Evaluation")
        test_results = self.evaluate(self.test_loader, "Final Test Evaluation")
        
        # Compute detailed metrics for test set
        test_detailed = self.compute_detailed_metrics(
            test_results['predictions'], 
            test_results['labels'],
            class_names
        )
        
        # Save results
        results = {
            'train_accuracy': train_results['accuracy'],
            'val_accuracy': val_results['accuracy'],
            'test_accuracy': test_results['accuracy'],
            'test_detailed_metrics': {
                'accuracy': test_detailed['accuracy'],
                'weighted_precision': test_detailed['weighted_precision'],
                'weighted_recall': test_detailed['weighted_recall'],
                'weighted_f1': test_detailed['weighted_f1']
            }
        }
        
        # Plot and save visualizations
        self.plot_training_curves(self.save_dir / 'training_curves.png')
        
        if class_names:
            self.plot_confusion_matrix(
                test_detailed['confusion_matrix'], 
                class_names,
                self.save_dir / 'confusion_matrix.png'
            )
        
        # Save detailed results
        with open(self.save_dir / 'final_results.json', 'w') as f:
            json.dump({
                'train_accuracy': float(train_results['accuracy']),
                'val_accuracy': float(val_results['accuracy']),
                'test_accuracy': float(test_results['accuracy']),
                'weighted_precision': float(test_detailed['weighted_precision']),
                'weighted_recall': float(test_detailed['weighted_recall']),
                'weighted_f1': float(test_detailed['weighted_f1'])
            }, f, indent=2)
        
        # Save classification report
        with open(self.save_dir / 'classification_report.txt', 'w') as f:
            f.write(test_detailed['classification_report'])
        
        print(f"\nFinal Results:")
        print(f"Train Accuracy: {train_results['accuracy']:.4f}")
        print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test F1-Score: {test_detailed['weighted_f1']:.4f}")
        
        return results